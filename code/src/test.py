import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from featurework import prepare_test_data, DATA_DIR, MODEL_DIR, OUTPUT_DIR


def read_csv_auto_encoding(file_path):
    """自动检测编码读取CSV"""
    encodings = ['utf-8', 'gbk', 'gb2312', 'latin1']
    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(file_path, encoding='utf-8', errors='replace')


def calculate_top_f1(predicted_max_stocks, predicted_min_stocks, check_file):
    """
    计算Top10最大涨幅和Top10最小涨幅的F1分数
    """
    try:
        check_df = read_csv_auto_encoding(check_file)

        # 获取预测的股票列表
        predicted_max = [str(x) for x in predicted_max_stocks]
        predicted_min = [str(x) for x in predicted_min_stocks]

        print(f"预测的涨幅最大Top10: {predicted_max}")
        print(f"预测的涨幅最小Top10: {predicted_min}")

        # 获取真实的Top10股票
        if '涨幅最大股票代码' in check_df.columns and '涨幅最小股票代码' in check_df.columns:
            # 如果check.csv有明确的分类
            true_max_stocks = check_df['涨幅最大股票代码'].dropna().astype(str).tolist()[:10]
            true_min_stocks = check_df['涨幅最小股票代码'].dropna().astype(str).tolist()[:10]
        elif '股票代码' in check_df.columns and '类型' in check_df.columns:
            # 如果check.csv有类型列
            true_max_stocks = check_df[check_df['类型'] == '最大']['股票代码'].dropna().astype(str).tolist()[:10]
            true_min_stocks = check_df[check_df['类型'] == '最小']['股票代码'].dropna().astype(str).tolist()[:10]
        else:
            # 假设前10行是最大涨幅，后10行是最小涨幅
            all_stocks = check_df.iloc[:, 0].dropna().astype(str).tolist()
            true_max_stocks = all_stocks[:10]
            true_min_stocks = all_stocks[10:20] if len(all_stocks) >= 20 else []

        print(f"真实的涨幅最大Top10: {true_max_stocks}")
        print(f"真实的涨幅最小Top10: {true_min_stocks}")

        # 计算最大涨幅Top10的F1
        max_correct = len(set(predicted_max) & set(true_max_stocks))
        max_precision = max_correct / 10 if len(predicted_max) > 0 else 0
        max_recall = max_correct / len(true_max_stocks) if len(true_max_stocks) > 0 else 0
        max_f1 = 2 * max_precision * max_recall / (max_precision + max_recall) if (
                                                                                              max_precision + max_recall) > 0 else 0

        # 计算最小涨幅Top10的F1
        min_correct = len(set(predicted_min) & set(true_min_stocks))
        min_precision = min_correct / 10 if len(predicted_min) > 0 else 0
        min_recall = min_correct / len(true_min_stocks) if len(true_min_stocks) > 0 else 0
        min_f1 = 2 * min_precision * min_recall / (min_precision + min_recall) if (
                                                                                              min_precision + min_recall) > 0 else 0

        # 计算综合F1（平均）
        avg_f1 = (max_f1 + min_f1) / 2

        print(f"\n涨幅最大Top10:")
        print(f"  正确预测数量: {max_correct}/10")
        print(f"  精确率: {max_precision:.4f}")
        print(f"  召回率: {max_recall:.4f}")
        print(f"  F1分数: {max_f1:.4f}")

        print(f"\n涨幅最小Top10:")
        print(f"  正确预测数量: {min_correct}/10")
        print(f"  精确率: {min_precision:.4f}")
        print(f"  召回率: {min_recall:.4f}")
        print(f"  F1分数: {min_f1:.4f}")

        print(f"\n综合F1分数: {avg_f1:.4f}")

        return {
            'max_correct': max_correct,
            'max_f1': max_f1,
            'min_correct': min_correct,
            'min_f1': min_f1,
            'avg_f1': avg_f1
        }

    except Exception as e:
        print(f"计算F1分数时出错: {e}")
        return {
            'max_correct': 0,
            'max_f1': 0,
            'min_correct': 0,
            'min_f1': 0,
            'avg_f1': 0
        }


def predict_future_returns():
    # 加载模型和预测
    model = joblib.load(MODEL_DIR / "lgb_optimized_model.pkl")
    df = read_csv_auto_encoding(DATA_DIR / "test.csv")
    df = prepare_test_data(df)

    # 预测
    features = [col for col in df.columns if col not in ['future_return', '日期', '股票代码', 'original_stock_code']]
    df['predicted_return'] = model.predict(df[features])

    # 获取Top 10股票
    last_records = df.sort_values(['original_stock_code', '日期']).groupby('original_stock_code').last()
    sorted_returns = last_records.sort_values('predicted_return', ascending=False)

    top_10_max = sorted_returns.head(10)['股票代码'].tolist()
    top_10_min = sorted_returns.tail(10)['股票代码'].tolist()

    # 保存预测结果
    result_df = pd.DataFrame({
        '涨幅最大股票代码': top_10_max,
        '涨幅最小股票代码': top_10_min
    })
    result_df.to_csv(OUTPUT_DIR / "result.csv", index=False)
    print(f"预测完成，结果保存到: {OUTPUT_DIR / 'result.csv'}")

    # 计算F1分数
    check_file = DATA_DIR / "check.csv"
    if check_file.exists():
        results = calculate_top_f1(top_10_max, top_10_min, check_file)

        # 保存评估结果
        with open(OUTPUT_DIR / "evaluation_results.txt", 'w', encoding='utf-8') as f:
            f.write("模型评估结果\n")
            f.write("=" * 40 + "\n")
            f.write(f"涨幅最大Top10:\n")
            f.write(f"  正确预测数量: {results['max_correct']}/10\n")
            f.write(f"  F1分数: {results['max_f1']:.4f}\n")
            f.write(f"涨幅最小Top10:\n")
            f.write(f"  正确预测数量: {results['min_correct']}/10\n")
            f.write(f"  F1分数: {results['min_f1']:.4f}\n")
            f.write(f"综合F1分数: {results['avg_f1']:.4f}\n")

        print(f"\n评估结果已保存到: {OUTPUT_DIR / 'evaluation_results.txt'}")
    else:
        print("未找到check.csv，跳过F1计算")


if __name__ == "__main__":
    predict_future_returns()