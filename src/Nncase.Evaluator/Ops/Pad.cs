using System;
using System.Linq;
using System.Collections.Generic;
using NetFabric.Hyperlinq;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using TorchSharp;

using torchF = TorchSharp.torch.nn.functional;

namespace Nncase.Evaluator.Ops
{
    public sealed partial class EvaluatorVisitor
    {
        private torch.Tensor VisitPad(Pad pad)
        {
            var input = _context.GetTorchArgument(pad, Pad.Input);
            var pads = _context.GetArgumentConst(pad, Pad.Pads).ToTensor<long>();
            var value = _context.GetArgumentConst(pad, Pad.Value).ToScalar<double>();
            List<long> torch_pads = new();
            for (int i = pads.Dimensions[0] - 1; i >= 0; i--)
            {
                torch_pads.Add(pads[i, 0]);
                torch_pads.Add(pads[i, 1]);
            }
            return torchF.pad(input, torch_pads.ToArray(), pad.PadMode.ToTorch(), value);
        }
    }
}