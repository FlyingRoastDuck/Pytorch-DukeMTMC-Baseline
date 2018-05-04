# **A Pytorch based Implementation of DukeMTMC-reID baseline(Resnet18,Resnet34 and Resnet50)**

## Introduction:
This project provides pytorch based implementation of Resnet18,Resnet34 and Resnet50. Their performance on DukeMTMC-reID are listed below (20 epochs):

|  Model Name   |   Rank 1 Acc  |      mAP      |
| ------------- |:-------------:|:-------------:|
|  Resnet 18    |     69.39%    |     45.90%    |
|  Resnet 34    |     74.10%    |     48.60%    |
|  Resnet 50    |     72.94%    |     48.12%    |

## Requirements:
**fire**,**pytorch**,**torchvision**,**numpy** are required to run this demo. 

**Pytorch  torchvision**

see http://pytorch.org/ for more information, you are supposed to install pytorch 0.3

**fire**

```angular2html
pip install fire
```

**numpy**
```angular2html
pip install numpy
```

**tqdm**
```angular2html
pip install tqdm
```

## Usage:
### 1. Train Model 

create 3 directories:dataReader/readyTrain dataReader/test and dataReader/query
and then execute download.sh to download images

change line 29 of config.py to select the model you want to train,the following models are supported:
- Resnet 50 ---- change self.model's value to "resnet50"
- Resnet 18 ---- change self.model's value to "resnet18"
- Resnet 34 ---- change self.model's value to "resnet34"

open a terminal in project's folder and run 
```
python reid.py train --modelPath=None
``` 
You can also change the model's training parameters with '--' in terminal, for example, if you want to change the initial learning rate of model to 0.01, you can run this command to start training:
```
python reid.py train --modelPath=None --lr=0.01
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

Thank for the guidance from [zhunzhong07](https://github.com/zhunzhong07) and [layumi](https://github.com/layumi)


## Log

- 2018.5.2

  1. update pytorch to 0.4.0, in 0.4.0, old code may have error.
  To avoid this, new version is now available in debug branch.
  2. tqdm is added to show a progress bar dynamicly.
  3. remove visdom support, there is no need to use visdom. I shall add it when necessary.

- 2018.5.3

  1. Code has been tested on market-1501. Results are listed below:

  |  Model Name   |   Rank 1 Acc  |      mAP      |
  | ------------- |:-------------:|:-------------:|
  |  Resnet 18    |     81.00%    |     50.42%    |
  |  Resnet 34    |       -       |       -       |
  |  Resnet 50    |       -       |       -       |
