#!/usr/bin/python
#-*-coding:utf-8-*-
#2015-03-09 Yuki Tomo

"""
input : yx_lists (3 or more)
output : curve.png
"""
import argparse
import sys
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

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
	switch = sys.argv[1] #PR or ROC
	output_file = sys.argv[2]
	test_results = sys.argv[3:] #複数のプロット用のx,yが格納されたファイル
	results_lists = []
	curve_names = []

	for test_result in test_results:
		directory, curve_name = os.path.split(test_result) #file_name の抽出
		curve_names.append(curve_name)
		y_elements, x_elements = load_submission(test_result)
		plt.plot(x_elements, y_elements)

	plt.xlim(0, 0.6)
	plt.ylim(0, 0.6)

	if switch == "PR":
		loc = 'upper right'
		title = 'PR curve'
		y_label = "Precision"
		x_label = "Recall"

	if switch == "ROC":
		loc = 'lower right'
		title = 'ROC curve'
		y_label = 'True Positive Rate'
		x_label = 'False Positive Rate'

	plt.title(title)
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.legend(curve_names, loc)
	#plt.show()
	plt.savefig(output_file)	

if __name__ == '__main__':
	main()
