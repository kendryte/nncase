from onnx.utils import extract_model
import argparse


def main(infile, outfile, inputs, outputs):
    if not isinstance(inputs, list):
        inputs = [inputs]
    if not isinstance(outputs, list):
        outputs = [outputs]
    extract_model(infile, outfile, inputs, outputs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Split ONNX Utils')
    parser.add_argument('infile')
    parser.add_argument('outfile')
    parser.add_argument('inputs')
    parser.add_argument('outputs')
    args = parser.parse_args()
    main(args.infile, args.outfile,
         args.inputs.split(','), args.outputs.split(','))
