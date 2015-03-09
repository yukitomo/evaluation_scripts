#!/usr/bin/python
#-*-coding:utf-8-*-
#2015-03-09 Yuki Tomo

"""
input : yx_lists (3 or more)
output : curve.png
"""
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
	output_file = sys.argv[1]
	test_results = sys.argv[2:] #複数のプロット用のx,yが格納されたファイル
	results_lists = []
	curve_names = []

	for test_result in test_results:
		directory, curve_name = os.path.split(test_result) #file_name の抽出
		curve_names.append(curve_name)
		y_elements, x_elements = load_submission(test_result)
		plt.plot(x_elements, y_elements)

	plt.xlim(0, 1)
	plt.ylim(0, 1)
	plt.legend(curve_names,'upper right')
	#plt.show()
	plt.savefig(output_file)	

if __name__ == '__main__':
	main()
