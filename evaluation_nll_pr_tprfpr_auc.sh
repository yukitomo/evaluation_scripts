#!/bin/shape
#sh evaluation_nll_pr_tprfpr_auc.sh 評価するファイル(拡張子はつけない)　評価するfileの保存場所 　アウトプットの保存先 
#example : $ sh evaluation_nll_pr_tprfpr_auc.sh from25to27_baseline . output_test

#evaluationするfile name(拡張子はつけない)
fileName=$1
#評価するfileの保存場所 
dirPath=$2
#outputの保存先
savePath=$3

#text
nllPath=${savePath}/nll_${fileName}.txt
prPath=${savePath}/pr_curve_${fileName}.txt
tprfprPath=${savePath}/tprfpr_curve_${fileName}.txt
aucPath=${savePath}/auc_${fileName}.txt

python calc_nll_pr_tprfpr_auc.py ${dirPath}/${fileName}.tsv ${nllPath} ${prPath} ${tprfprPath} ${aucPath}

#png
prgraphPath=${savePath}/pr_curve_${fileName}.png
tprfprgraphPath=${savePath}/tprfpr_curve_${fileName}.png

python curve_prot.py ${prPath} ${prgraphPath}
python curve_prot.py ${tprfprPath} ${tprfprgraphPath}