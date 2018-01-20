clear;clc;close all
cvAcc=load('cvAcc.txt');
lossVal=load('lossVal.txt');
trainAcc=load('trainAcc.txt');
plot(cvAcc,'r*');hold on;grid on
plot(trainAcc,'bo');xlabel('迭代次数')
ylabel('准确率');legend({'测试集精确度','训练集精确度'})