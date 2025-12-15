import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import joblib
import warnings
from featurework import prepare_train_data, DATA_DIR, MODEL_DIR
from sklearn.ensemble import VotingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
import optuna
from optuna.integration import LightGBMPruningCallback

warnings.filterwarnings('ignore')


class FeatureAugmenter(BaseEstimator, TransformerMixin):
    """Custom feature engineering transformer"""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Ensure we're working with a copy
        X = X.copy()

        # Price-based features
        X['close_open_diff'] = X['收盘'] - X['开盘']
        X['high_low_diff'] = X['最高'] - X['最低']
        X['close_ma5_ratio'] = X['收盘'] / X['ma5']
        X['volume_ma5_ratio'] = X['成交量'] / X['成交量'].rolling(5).mean()

        # Volatility features
        X['volatility_5'] = X['收盘'].pct_change().rolling(5).std()
        X['volatility_10'] = X['收盘'].pct_change().rolling(10).std()

        # Momentum features
        X['momentum_5_1'] = X['收盘'].pct_change(5) - X['收盘'].pct_change(1)
        X['momentum_10_5'] = X['收盘'].pct_change(10) - X['收盘'].pct_change(5)

        # Mean reversion features
        X['mean_reversion_5'] = (X['收盘'] - X['ma5']) / X['std5']
        X['mean_reversion_10'] = (X['收盘'] - X['ma10']) / X['std10']

        # Volume features
        X['volume_change'] = X['成交量'].pct_change()
        X['volume_price_corr'] = X['成交量'].rolling(10).corr(X['收盘'])

        # Fill any remaining NA values
        X.fillna(method='ffill', inplace=True)
        X.fillna(method='bfill', inplace=True)
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(0, inplace=True)

        return X


def objective(trial, X, y, tscv):
    """Optuna optimization objective function"""
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 1.0),
        'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 1.0),
        'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 0.1),
        'verbosity': -1,
        'seed': 42,
        'n_jobs': -1
    }

    scores = []

    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_index], X.iloc[test_index]
        y_train, y_val = y.iloc[train_index], y.iloc[test_index]

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],  # Explicitly name the validation sets
            verbose_eval=False,
            callbacks=[LightGBMPruningCallback(trial, 'rmse', valid_name='valid')]  # Match the valid_name here
        )

        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        scores.append(rmse)

    return np.mean(scores)


def train_ensemble_model(X, y):
    """Train an ensemble of models"""
    # LightGBM model
    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'learning_rate': 0.01,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'max_depth': 8,
        'min_data_in_leaf': 30,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'min_split_gain': 0.01,
        'verbosity': -1,
        'n_jobs': -1,
        'seed': 42
    }

    # XGBoost model
    xgb_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'eta': 0.01,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'lambda': 0.1,
        'alpha': 0.1,
        'n_jobs': -1,
        'seed': 42
    }

    # Create pipeline with feature augmentation and scaling
    pipeline = make_pipeline(
        FeatureAugmenter(),
        RobustScaler(),
        PCA(n_components=0.95),  # Keep 95% of variance
        VotingRegressor([
            ('lgbm', lgb.LGBMRegressor(**lgb_params, n_estimators=1000)),
            ('xgb', xgb.XGBRegressor(**xgb_params, n_estimators=1000))
        ])
    )

    # Train the pipeline
    pipeline.fit(X, y)

    return pipeline


def train_model():
    # Read training data
    train_file = DATA_DIR / "train.csv"
    df = pd.read_csv(train_file)

    # Prepare training data
    df = prepare_train_data(df)

    # Clean and prepare data
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df.dropna(inplace=True)

    # Define features and target
    features = [col for col in df.columns if col not in ['future_return', '日期', '股票代码', 'original_stock_code']]
    X = df[features]
    y = df['future_return']

    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=3)

    # Optuna optimization
    print("Starting hyperparameter optimization...")
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X, y, tscv), n_trials=30)

    print("Best trial:")
    trial = study.best_trial
    print(f"  RMSE: {trial.value:.4f}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Train final models with best parameters
    print("\nTraining final models...")

    # 1. Train optimized LightGBM model
    lgb_params = trial.params
    lgb_params.update({
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'n_jobs': -1,
        'seed': 42
    })

    lgb_model = lgb.LGBMRegressor(**lgb_params, n_estimators=1000)
    lgb_model.fit(X, y)

    # 2. Train ensemble model
    ensemble_model = train_ensemble_model(X, y)

    # Save models
    joblib.dump(lgb_model, MODEL_DIR / "lgb_optimized_model.pkl")
    joblib.dump(ensemble_model, MODEL_DIR / "ensemble_model.pkl")

    print("\nModels saved successfully.")


if __name__ == "__main__":
    train_model()