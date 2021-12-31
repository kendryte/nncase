using System;
using System.Linq;
using System.Collections.Generic;
using NetFabric.Hyperlinq;
using Nncase.IR.NN;
using TorchSharp;

using torchF = TorchSharp.torch.nn.functional;

namespace Nncase.Evaluator.Ops
{
    public sealed partial class EvaluatorVisitor
    {
        private torch.Tensor VisitConv2DTranspose(Conv2DTranspose conv)
        {
            var input = _context.GetTorchArgument(conv, Conv2DTranspose.Input);
            var weights = _context.GetTorchArgument(conv, Conv2DTranspose.Weights);
            var bias = _context.GetTorchArgument(conv, Conv2DTranspose.Bias);
            var stride = _context.GetArgumentConst(conv, Conv2DTranspose.Stride).ToArray<long>();
            // [w:[left right] h:[top bottom]]
            var pad = _context.GetArgumentConst(conv, Conv2DTranspose.Padding).ToTensor<long>();
            var dilation = _context.GetArgumentConst(conv, Conv2DTranspose.Dilation).ToArray<long>();
            var groups = _context.GetArgumentConst(conv, Conv2DTranspose.Groups).ToScalar<long>();
            if (conv.PadMode != PadMode.Constant)
            {
                throw new NotImplementedException($"Conv2DTranspose with {conv.PadMode}!");
            }

            // var afterPad = torchF.pad(input, new long[] { pad[0, 0], pad[1, 0], pad[0, 1], pad[1, 1] });
            return torchF.conv_transpose2d(
                input,
              weights, bias, 
                stride,
              dilation: dilation,
              groups: groups);
        }
    }
}