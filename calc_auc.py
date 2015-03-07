#!/usr/bin/python
#-*-coding:utf-8-*-
#2015-03-07 Yuki Tomo

import numpy as np
import sys

"""
input : y_list, x_list
output : area under curve 
"""
def load_submission(file_path):
    y_elements, x_elements = [], []
    with open(file_path, 'r') as f:
        for line in f:
            elems = line.strip().replace("\t", " ").split(" ")
            y = float(elems[0])
            x = float(elems[1])
            y_elements.append(y)
            x_elements.append(x)
    return np.array(y_elements), np.array(x_elements)

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
    #x, y = check_arrays(x, y)
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



def main():
	y_elements, x_elements = load_submission(sys.argv[1])
	print "auc_score : ",auc(x_elements, y_elements, reorder=True) 

if __name__ == '__main__':
	main()