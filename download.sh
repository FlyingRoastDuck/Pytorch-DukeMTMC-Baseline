#!/bin/sh
#获得文件
zipfile="duke.zip"
if [ ! -f $zipfile ]; then
    echo "Downloading..."
    wget -c --referer=https://pan.baidu.com/s/1kUD80xp -O duke.zip "https://d.pcs.baidu.com/file/d09cd997d6f37f0ed98fb7682606bcc4?fid=1312005134-250528-595029897752223&time=1515038694&rt=sh&sign=FDTAERV-DCb740ccc5511e5e8fedcff06b081203-yZu5klFC%2Bw382u0iK1u9xnWYA%2FM%3D&expires=8h&chkv=1&chkbd=0&chkpc=&dp-logid=81253806272606200&dp-callid=0&r=565593815"
fi
#解压、覆盖
unzip -oj duke.zip "DukeMTMC-reID/bounding_box_train/*" -d dataReader/readyTrain/
unzip -oj duke.zip "DukeMTMC-reID/bounding_box_test/*" -d dataReader/test/
unzip -oj duke.zip "DukeMTMC-reID/query/*" -d dataReader/query/
echo "Done"