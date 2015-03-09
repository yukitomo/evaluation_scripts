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