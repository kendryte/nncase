from bokeh.io import export_svgs
from bokeh.layouts import column
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, show, save, output_file
from pandas import *
import getopt
import math
import sys

def show_quant_error(argv):
    quant_error_file = ""

    opts, args = getopt.getopt(argv,"hq:",["quant_error_file="])
    if opts == []:
        print("python show_quant_error.py -q <quant_error_file>")
        sys.exit()

    for opt, arg in opts:
        if opt == '-h':
            print("python show_quant_error.py --quant_error_file <quant_error_file>")
            sys.exit()
        elif opt in ("-q", "--quant_error_file"):
            quant_error_file = arg
        else:
            print("python show_quant_error.py --quant_error_file <quant_error_file>")
            sys.exit()

    f = read_csv(quant_error_file)
    cosine_error_list = f[' cosine_error'].tolist()
    mre_list = f[' mre_error'].tolist()
    layer_len = len(cosine_error_list)

    x = list(range(layer_len))
    desc = f['name'].tolist()

    source_cosine = ColumnDataSource(data = dict(
        x = x,
        y = cosine_error_list,
        desc = desc,
    ))
    source_mre = ColumnDataSource(data = dict(
        x = x,
        y = mre_list,
        desc = desc,
    ))

    TOOLTIPS_COSINE = [
        ("desc", "@desc"),
        ("cosine", "@y"),
    ]

    TOOLTIPS_MRE = [
        ("desc", "@desc"),
        ("mre", "@y"),
    ]

    p_cosine = figure(x_range=desc, width = 2000, height = 500, tooltips = TOOLTIPS_COSINE, title = "cosine", y_range=(0, 1))
    p_cosine.circle('x', 'y', size = 5, source = source_cosine, color = "red")
    p_mre = figure(x_range=p_cosine.x_range, width = 2000, height = 500, tooltips = TOOLTIPS_MRE, title = "mre")
    p_mre.circle('x', 'y', size = 5, source = source_mre, color = "blue")

    p_cosine.xaxis.major_label_orientation = math.pi / 2
    p_cosine.xaxis.visible = False
    p_mre.xaxis.major_label_orientation = math.pi / 2
    p_mre.xaxis.visible = False

    p_cosine.output_backend = "svg"
    p_mre.output_backend = "svg"

    output_file('plot.html', mode='inline')
    save(gridplot([p_cosine, p_mre], ncols = 1, toolbar_location = 'right'))
    #show(gridplot([[p_cosine, p_mre]], toolbar_location = 'right'))
    #show(column(p_cosine, p_mre))

    export_svgs(p_cosine, filename = "p_cosine.svg")
    export_svgs(p_mre, filename = "p_mre.svg")

if __name__ == "__main__":
    show_quant_error(sys.argv[1:])
