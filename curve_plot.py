#!/usr/bin/python
#-*-coding:utf-8-*-
#2015-03-04 Yuki Tomo

#$ python curve_prot.py  pr_curve.txt output.png

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load_submission(file_path):
    y_elements, x_elements = [], []
    with open(file_path, 'r') as f:
        for line in f:
            elems = line.strip().split(" ")
            y = float(elems[0])
            x = float(elems[1])
            y_elements.append(y)
            x_elements.append(x)
    return y_elements, x_elements

def main():
    y_x_file = sys.argv[1] #[precision, recall], [tpr, fpr]
    output_file = sys.argv[2]

    y_elements, x_elements = load_submission(y_x_file) 
    plt.plot(x_elements, y_elements)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    #plt.show()
    plt.savefig(output_file)

if __name__ == '__main__':
	main()