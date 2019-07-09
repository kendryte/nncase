using System;
using System.Collections.Generic;
using System.Numerics.Tensors;
using System.Text;
using NnCase.IR;
using NnCase.IR.Operators;

namespace NnCase.Importer
{
    /// <summary>
    /// TFLite importer convolution ops lowering.
    /// </summary>
    public partial class TFLiteImporter
    {
        private void ConvertFullyConnected(tflite.Operator op)
        {
            var inputA = GetTensor(op.Inputs(0));
            var inputB = GetTensor(op.Inputs(1));
            var bias = GetTensor(op.Inputs(2));
            var options = op.BuiltinOptions<tflite.FullyConnectedOptions>().Value;

            if (options.WeightsFormat != tflite.FullyConnectedOptionsWeightsFormat.DEFAULT)
                throw new NotSupportedException($"Unsupported fullyConnected weights format: {options.WeightsFormat}");

            var biasTensor = LoadTensor<float>(bias);
            var reshape = _graph.AddNode(new Reshape(DataType.Float32, GetShape(inputA), new[] { -1, GetShape(inputB)[1] }));
            var inputBTp = _graph.AddNode(new Transpose(DataType.Float32, GetShape(inputB), new[] { 1, 0 }));
            var matmul = _graph.AddNode(new MatMul(reshape.Output.Shape, inputBTp.Output.Shape, biasTensor, ToFloatClampRange(options.FusedActivationFunction)));
            matmul.InputA.Connect(reshape.Output);
            matmul.InputB.Connect(inputBTp.Output);

            _inputTensors.Add(reshape.Input, op.Inputs(0));
            _inputTensors.Add(inputBTp.Input, op.Inputs(1));
            _outputTensors.Add(op.Outputs(0), matmul.Output);
        }
    }
}
