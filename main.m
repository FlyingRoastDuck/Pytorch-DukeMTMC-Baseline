clear;clc;close all
cvAcc=load('cvAcc.txt');
lossVal=load('lossVal.txt');
trainAcc=load('trainAcc.txt');
plot(cvAcc,'r*');hold on;grid on
plot(trainAcc,'bo');xlabel('��������')
ylabel('׼ȷ��');legend({'���Լ���ȷ��','ѵ������ȷ��'})