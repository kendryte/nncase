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
        private torch.Tensor VisitConv2D(Conv2D conv)
        {
            var input = _context.GetArgument(conv, Conv2D.Input);
            var weights = _context.GetArgument(conv, Conv2D.Weights);
            var bias = _context.GetArgument(conv, Conv2D.Bias);
            var stride = _context.GetArgumentConst(conv, Conv2D.Stride).ToTensor<long>();
            var pad = _context.GetArgumentConst(conv, Conv2D.Padding).ToTensor<long>();
            var dilation = _context.GetArgumentConst(conv, Conv2D.Dilation).ToTensor<long>();
            var groups = _context.GetArgumentConst(conv, Conv2D.Groups).ToScalar<long>();
            if (conv.PadMode != PadMode.Constant)
            {
                throw new NotImplementedException($"Conv2D with {conv.PadMode}!");
            }
            return torchF.conv2d(
              torchF.pad(input, new long[] { pad[1, 0], pad[1, 1], pad[0, 0], pad[0, 1] }),
              weights, bias,
              strides: new long[] { stride[0], stride[1] },
              dilation: new long[] { dilation[0], dilation[1] },
              groups: groups);
        }
    }
}