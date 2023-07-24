from bokeh.io import export_svgs
from bokeh.layouts import column
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, show, save, output_file
from matplotlib import pyplot as plt
from pandas import *
from pathlib import Path
import getopt
import math
import numpy as np
import sys

def show_quant_data_histogram(argv):
    dump_tensor_path = ""

    opts, args = getopt.getopt(argv,"hd:",["dump_tensor_path="])
    if opts == []:
        print("python show_quant_data_histogram.py -d <dump_tensor_path>")
        sys.exit()

    for opt, arg in opts:
        if opt == '-h':
            print("python show_quant_data_histogram.py --dump_tensor_path <dump_tensor_path>")
            sys.exit()
        elif opt in ("-d", "--dump_tensor_path"):
            dump_tensor_path = arg
        else:
            print("python show_quant_data_histogram.py --dump_tensor_path <dump_tensor_path>")
            sys.exit()

    # print(dump_tensor_path)

    pathlist = Path(dump_tensor_path).glob('**/*.csv')
    for path in pathlist:
        path_in_str = str(path)
        # print(path_in_str)
        f = read_csv(path_in_str)
        gt_list = np.array(f['ground_truth'].tolist())
        simulate_list = np.array(f[' simulate_output'].tolist())
        # print(np.array(f['ground_truth'].tolist()))
        # draw histogram here
        plt.hist(gt_list, alpha = 0.5, bins = 50, color = "blue", label = 'gt')
        plt.hist(simulate_list, alpha = 0.5, bins = 50, color = "green", label = 'simulate')
        # plt.title(path_in_str)
        plt.legend()
        # plt.show()
        plt.savefig(path_in_str[:-3] + "svg")
        plt.clf()
        # exit()

if __name__ == "__main__":
    show_quant_data_histogram(sys.argv[1:])
