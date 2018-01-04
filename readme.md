# **A Pytorch based Implementation of DukeMTMC-reID baseline(AlexNet,ResNet18,Resnet34)**

## Introduction:
This project provides pytorch based implementation of Alexnet,Resnet18 and Resnet34. Their performance on DukeMTMC-reID are listed below:

|Model Name     |    Rank 1 Acc |
| ------------- |:-------------:|
|AlexNet        |      40%      |
|Resnet 18      |     52.11%    |
|Resnet 34      |     59.68%    |

## Requirements:
**fire**,**pytorch**,**torchvision**,**numpy**,**matplotlib** are required to run this demo. You can install [Anaconda](https://www.anaconda.com/download/) or [Miniconda](https://conda.io/miniconda.html) to 
reduce your configuration time


## Usage:
### 1. Train Model 
run download.sh to download the dataset
change line 29 of config.py to select the model you want to train,the following models are supported:
- Alexnet ---- change self.model's value to "AlexNet"
- Resnet 18 ---- change self.model's value to "resnet18"
- Resnet 34 ---- change self.model's value to "resnet34"

open a terminal in project's folder and run 
```
python reid.py train --modelPath=None
``` 
You can also change the model's training parameters with '--' in terminal, for example, if you want to change the initial learning rate of model to 1e-5, you can run this command to start training:
```
python reid.py train --modelPath=None --lr=1e-5
```
During the training process, you will get your model's weight file called ($modelname+$time).pth in 'snapshots' folder (please create it) in every 10 epoch. 
You can change the frequency in line 31 of config.py(self.snapFreq)

**warning:** if you want to train the model with target pth file, you may change the modelPath value in config.py and run the command above without '--modelPath=None'.

### 2.  Get Features(Test)
After training, run this command to get allF.pth, which stores all features of images in 'dataReader\test\'
```
python reid.py test
```

### 3. Query your Image
If you want to query target image ,just check the serial number of the image(for example, 0005_c2_f0046985.jpg is the first image in 'dataReader\query\' so its serial number is 0) in query image folder and run this command in terminal
```
python reid.py query 0
```
If you run command like this
```
python reid.py query
```
all images in 'dataReader\query\' will be queried and Rank1-6 values will be listed afterwards.

## Reference and Acknowledgment:
I referenced the code on https://github.com/chenyuntc/pytorch-best-practice and I think it's the best tutorial code for pytorch beginners

Thank for the guidance from [Zhun Zhong](https://github.com/zhunzhong07)
