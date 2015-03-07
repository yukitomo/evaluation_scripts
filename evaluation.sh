#!/bin/sh
 
fileName=$1
dirPath=$2
gzipName=${fileName}.gz
 
# gzip形式のファイルをDownload、展開
curl -blogin.cookie -o ${gzipName} "https://tameike.fout.jp/filebrowser/download//user/kitsune/${dirPath}"
gzip -d ${gzipName}
 
# 制御文字をタブに置換
sed -i -e 's/^\([1\|0\]\.0\)[^0-9]*\([0-9]\)/\1\t\2/' ./${fileName}
# 検定
#python calculate_pr_curve_and_nll.py ${fileName} pr_curve.txt nll.txt

