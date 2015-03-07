#!/usr/bin/python
#-*-coding:utf-8-*-
#2015-03-06 Yuki Tomo
#評価手法：negative LogLoss, Precision-Recall(PR)-curve, ROC-curve 

"""
calc AUC(FPr_list, TPr_list), nll(labels, probs)
output ROC(FPr_list, TPr_list), PRC(Recall_list, Precsion_list)   
"""

import argparse
import warnings
import numpy as np
import pylab as plt
import sys
from math import exp,log

def _assert_all_finite(X):
    """Like assert_all_finite, but only for ndarray.""" 
    if (X.dtype.char in np.typecodes['AllFloat'] and not np.isfinite(X.sum())
            and not np.isfinite(X).all()):
        raise ValueError("Array contains NaN or infinity.")

def _num_samples(x):
    """Return number of samples in array-like x.""" 
    if not hasattr(x, '__len__') and not hasattr(x, 'shape'):
        raise TypeError("Expected sequence or array-like, got %r" % x)
    return x.shape[0] if hasattr(x, 'shape') else len(x)

def check_arrays(*arrays, **options):
	"""Check that all arrays have consistent first dimensions.

	Checks whether all objects in arrays have the same shape or length.
	By default lists and tuples are converted to numpy arrays.

	It is possible to enforce certain properties, such as dtype, continguity
	and sparse matrix format (if a sparse matrix is passed).

	Converting lists to arrays can be disabled by setting ``allow_lists=True``.
	Lists can then contain arbitrary objects and are not checked for dtype,
	finiteness or anything else but length. Arrays are still checked
	and possibly converted.

	Parameters
	----------
	*arrays : sequence of arrays or scipy.sparse matrices with same shape[0]
		Python lists or tuples occurring in arrays are converted to 1D numpy
		arrays, unless allow_lists is specified.

	sparse_format : 'csr', 'csc' or 'dense', None by default
		If not None, any scipy.sparse matrix is converted to
		Compressed Sparse Rows or Compressed Sparse Columns representations.
		If 'dense', an error is raised when a sparse array is
		passed.

	copy : boolean, False by default
		If copy is True, ensure that returned arrays are copies of the original
		(if not already converted to another format earlier in the process).

	check_ccontiguous : boolean, False by default
		Check that the arrays are C contiguous

	dtype : a numpy dtype instance, None by default
		Enforce a specific dtype.

	allow_lists : bool
		Allow lists of arbitrary objects as input, just check their length.
		Disables
	""" 
	sparse_format = options.pop('sparse_format', None)
	if sparse_format not in (None, 'csr', 'csc', 'dense'):
		raise ValueError('Unexpected sparse format: %r' % sparse_format)
	copy = options.pop('copy', False)
	check_ccontiguous = options.pop('check_ccontiguous', False)
	dtype = options.pop('dtype', None)
	allow_lists = options.pop('allow_lists', False)
	if options:
		raise TypeError("Unexpected keyword arguments: %r" % options.keys())

	if len(arrays) == 0:
		return None

	n_samples = _num_samples(arrays[0])

	checked_arrays = []
	for array in arrays:
		array_orig = array
		if array is None:
			# special case: ignore optional y=None kwarg pattern
			checked_arrays.append(array)
			continue
		size = _num_samples(array)

		if size != n_samples:
			raise ValueError("Found array with dim %d. Expected %d" 
							 % (size, n_samples))

		if not allow_lists or hasattr(array, "shape"):
			if check_ccontiguous:
				array = np.ascontiguousarray(array, dtype=dtype)
			else:
				array = np.asarray(array, dtype=dtype)
			_assert_all_finite(array)

		if copy and array is array_orig:
			array = array.copy()
		checked_arrays.append(array)

	return checked_arrays

def auc(x, y, reorder=False):
	"""Compute Area Under the Curve (AUC) using the trapezoidal rule

	This is a general function, given points on a curve.  For computing the
	area under the ROC-curve, see :func:`roc_auc_score`.

	Parameters
	----------
	x : array, shape = [n]
		x coordinates.

	y : array, shape = [n]
		y coordinates.

	reorder : boolean, optional (default=False)
		If True, assume that the curve is ascending in the case of ties, as for
		an ROC curve. If the curve is non-ascending, the result will be wrong.

	Returns
	-------
	auc : float

	Examples
	--------
	>>> import numpy as np
	>>> from sklearn import metrics
	>>> y = np.array([1, 1, 2, 2])
	>>> pred = np.array([0.1, 0.4, 0.35, 0.8])
	>>> fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=2)
	>>> metrics.auc(fpr, tpr)
	0.75

	See also
	--------
	roc_auc_score : Computes the area under the ROC curve

	precision_recall_curve :
		Compute precision-recall pairs for different probability thresholds

	""" 
	x, y = check_arrays(x, y)
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

"""
def parse_args():
	p = argparse.ArgumentParser(description='calc auc')
	p.add_argument('-s', '--submission', type=str, required=True) #input_file
	p.add_argument('-a', action='store_true', default=False, dest = 'auc_switch', help='calc AUC prob')
	p.add_argument('-r', action='store_true', default=False, dest = 'roc_switch', help='plot ROC graph')
	return p.parse_args() 
"""

def main():
	
	labels_probs_file_path = sys.argv[1]
	nll_file_path = sys.argv[2]
	pr_file_path = sys.argv[3]
	tpr_fpr_file_path = sys.argv[4]
	auc_prc_roc_file_path = sys.argv[5]

	labels, probs = load_submission(labels_probs_file_path)

	#calculate Negative Log Likelihood
	avg_nll_all, avg_nll_pos, avg_nll_neg = calc_nlls(labels, probs)
	nll_file = open(nll_file_path,"wb")
	nll_file.write('the average negative log likelihood on all samplesis: %f\n'%(avg_nll_all))
	nll_file.write('the average negative log likelihood on positive samples is: %f\n'%(avg_nll_pos))
	nll_file.write('the average negative log likelihood on negative samples is: %f\n'%(avg_nll_neg))

	#calculate Precision, Recall, fp_rates, tp_rates　を計算してファイルに書き込み
	recalls, precisions, fp_rates, tp_rates = calc_pr_fprtpr(labels, probs, pr_file_path, tpr_fpr_file_path)

	pr_auc = auc(recalls, precisions, reorder=True)
	roc_auc = auc(fp_rates, tp_rates, reorder=True)

	auc_file = open(auc_prc_roc_file_path,"wb")
	auc_file.write("pr_auc\t%f\n"%pr_auc)
	auc_file.write("roc_auc\t%f\n"%roc_auc)


if __name__ == '__main__':
	main()