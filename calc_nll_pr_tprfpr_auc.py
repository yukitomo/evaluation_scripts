#!/usr/bin/python
#-*-coding:utf-8-*-
#評価手法：negative LogLoss, Precision-Recall(PR)-curve, ROC-curve, auPRC, auROC

import numpy as np
import sys
from math import exp,log

def auc(x, y, reorder=False):

	if x.shape[0] < 2:
		raise ValueError('At least 2 points are needed to compute'
						 ' area under curve, but x.shape = %s' % x.shape)

	direction = 1
	if reorder:
		# reorder the data points according to the x axis and using y to
		# break ties
		order = np.lexsort((y, x))
		x, y = x[order], y[order]
	else:
		dx = np.diff(x)
		if np.any(dx < 0):
			if np.all(dx <= 0):
				direction = -1
			else:
				raise ValueError("Reordering is not turned on, and " 
								 "the x array is not increasing: %s" % x)

	area = direction * np.trapz(y, x)

	return area

def make_pn_accumulate(labels, probs):
	index = sorted(range(len(probs)), key=lambda k: probs[k])
	positive_accumulate = [0] * len(probs)
	negative_accumulate = [0] * len(probs)

	if labels[index[0]] == 1:
		positive_accumulate[0] += 1
	else:
		negative_accumulate[0] += 1

	for i in range(1,len(probs)):
		ind = index[i]
		if labels[ind] == 1:
			positive_accumulate[i] = positive_accumulate[i-1]+1
			negative_accumulate[i] = negative_accumulate[i-1]
		else:
			positive_accumulate[i] = positive_accumulate[i-1]
			negative_accumulate[i] = negative_accumulate[i-1]+1
	
	positive_number = positive_accumulate[-1]
	negative_number = negative_accumulate[-1]

	return positive_accumulate, negative_accumulate, positive_number, negative_number

def calc_pr_fprtpr(labels, probs, pr_file_path, tpr_fpr_file_path):
	pr_file = open(pr_file_path, 'wb')
	tpr_fpr_file_path = open(tpr_fpr_file_path, 'wb')

	positive_accumulate, negative_accumulate, positive_number, negative_number = make_pn_accumulate(labels, probs)

	recalls = []
	precisions = []
	fp_rates = []
	tp_rates = []

	for i in range(1,len(probs)-1):
		true_positive = positive_number - positive_accumulate[i]
		false_positive = negative_number - negative_accumulate[i]
		true_negative = negative_accumulate[i]
		false_negative = positive_accumulate[i]

		recall = 1.0 * true_positive / (true_positive + false_negative)
		precision = 1.0 * true_positive / (true_positive + false_positive)
		pr_file.write(str(precision)+' '+str(recall)+'\n')
		recalls.append(recall)
		precisions.append(precision)

		fp_rate = 1.0 * false_positive / (false_positive + true_negative)
		tp_rate = recall
		tpr_fpr_file_path.write(str(tp_rate)+' '+str(fp_rate)+'\n')
		fp_rates.append(fp_rate)
		tp_rates.append(tp_rate)

	return np.array(recalls), np.array(precisions), np.array(fp_rates), np.array(tp_rates)



def log_loss(label, prob):
	return - log(prob) if label > 0 else -log(1 - prob)

def calc_nlls(labels, probs):
	_nll = [0.0]*2
	_num = [0]*2
	for i in range(0,len(labels)):
		_nll[labels[i]] += log_loss(labels[i], probs[i])
		_num[labels[i]] += 1

	#the average negative log likelihood on all samplesis:
	avg_nll_all = 1.0 * (_nll[0] + _nll[1]) / (_num[0] + _num[1])
	#the average negative log likelihood on positive samplesis:
	avg_nll_pos = 1.0 * _nll[1] / _num[1]
	#the average negative log likelihood on negative samples
	avg_nll_neg = 1.0 * _nll[0]/_num[0]

	return avg_nll_all, avg_nll_pos, avg_nll_neg


def load_submission(file_path):
	labels, probs = [], []
	with open(file_path, 'r') as f:
		for line in f:
			elems = line.strip().replace("\t"," ").split(" ")
			label = int(float(elems[0]))
			prob = float(elems[1])
			labels.append(label)
			probs.append(prob)
	return np.array(labels), np.array(probs) 


def main():
	
	labels_probs_file_path = sys.argv[1] #input_file : labele, probability 
	nll_auc_file_path = sys.argv[2] #output_file : nll, nll(P), nll(n), auPRC, auROC 
	pr_file_path = sys.argv[3] #output_file : PRcurve 
	tpr_fpr_file_path = sys.argv[4] #output_file : ROCcurve

	labels, probs = load_submission(labels_probs_file_path)
	avg_nll_all, avg_nll_pos, avg_nll_neg = calc_nlls(labels, probs)
	
	#calculate Precision, Recall, fp_rates, tp_rates
	recalls, precisions, fp_rates, tp_rates = calc_pr_fprtpr(labels, probs, pr_file_path, tpr_fpr_file_path)

	pr_auc = auc(recalls, precisions, reorder=True)
	roc_auc = auc(fp_rates, tp_rates, reorder=True)

	nll_auc_file = open(nll_auc_file_path,"wb")
	nll_auc_file.write("nll\tnll(p)\tnll(n)\tauROC\tauPRC\n") 
	nll_auc_file.write("%f\t%f\t%f\t%f\t%f\n"%(avg_nll_all, avg_nll_pos, avg_nll_neg, roc_auc, pr_auc))
	nll_auc_file.write("\n")

	nll_auc_file.write("%f\tavg_nll_all\n"%avg_nll_all)
	nll_auc_file.write("%f\troc_auc\n"%roc_auc)
	nll_auc_file.write("%f\tprc_auc\n"%pr_auc)


if __name__ == '__main__':
	main()