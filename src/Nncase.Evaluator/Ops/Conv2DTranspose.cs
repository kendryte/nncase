using System;
using System.Linq;
using System.Collections.Generic;
using NetFabric.Hyperlinq;
using Nncase.IR.NN;
using TorchSharp;
using Nncase.IR;

using torchF = TorchSharp.torch.nn.functional;

namespace Nncase.Evaluator.Ops
{
    public class Conv2DTransposeEvaluator : IEvaluator<Conv2DTranspose>
    {
        public static Const Visit(EvaluatorContext context, Conv2DTranspose conv)
        {
            var input = context.GetTorchArgument(conv, Conv2DTranspose.Input);
            var weights = context.GetTorchArgument(conv, Conv2DTranspose.Weights);
            var bias = context.GetTorchArgument(conv, Conv2DTranspose.Bias);
            var stride = context.GetArgumentConst(conv, Conv2DTranspose.Stride).ToArray<long>();
            // [w:[left right] h:[top bottom]]
            var pad = context.GetArgumentConst(conv, Conv2DTranspose.Padding).ToTensor<long>();
            var dilation = context.GetArgumentConst(conv, Conv2DTranspose.Dilation).ToArray<long>();
            var groups = context.GetArgumentConst(conv, Conv2DTranspose.Groups).ToScalar<long>();
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
              groups: groups).ToConst();
        }
    }
}