## Baseline说明

文件夹顺序如下

```
|--project
    |--code（储存运行代码，docker内）
        |--src
            |--featurework.py
            |--test.py
            |--train.py
    |--data（储存数据，在docker-compose.yml中进行挂载）
        |--test.csv
        |--train.csv
    |--model（储存训练模型，docker内）
    |--output（储存计算结果，在docker-compose.yml中进行挂载）
        |--reslut.csv
    |--temp（储存中间结果，在docker-compose.yml中进行挂载）
    |--init.sh（docker内）
    |--train.sh（docker内）
    |--test.sh（docker内）
    |--docker-compose.yml
    |--run.sh（docker外）
```

在测评时通过init.sh进行初始化，train.sh进行训练，test.sh进行推理，score.sh进行打分，该代码提供给选手参考，选手可以根据自己的需求进行修改，选手可以在code文件夹中编写对应代码或自行创建代码，同时需将训练出来的文件存储在model文件夹内，将输出结果存储在output文件夹内，计算的中间结果可以储存在temp文件夹内

注意，提交时需要提交一个打包好的docker，选手需要按照上面所示将文件进行存储，在docker中储存code，model，init.sh，train.sh，test.sh等文件，将docker镜像命名为bdc2025，并导出为队伍名称.tar进行提交，我们会使用docker load -i 队伍名称.tar命令对docker镜像进行加载，并使用下发的docker-compose.yml文件运行。

在进行结果计算时，我们会将run.sh修改为如下命令，使用docker-compose up进行计算，此时需要保证所训练的模型已储存在model文件夹中

```sh
/bin/bash /app/init.sh
/bin/bash /app/test.sh
```

在进行程序复现时，我们会将run.sh修改为如下命令，使用docker-compose up进行计算

```sh
/bin/bash /app/init.sh
/bin/bash /app/train.sh
/bin/bash /app/test.sh
```

