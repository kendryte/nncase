using System;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using TorchSharp;

using torchF = TorchSharp.torch.nn.functional;
namespace Nncase.Evaluator.Ops
{
    public sealed partial class EvaluatorVisitor
    {
        private torch.Tensor VisitClamp(Clamp clamp)
        {
            var input = _context.GetArgument(clamp, Clamp.Input);
            var min = _context.GetArgumentConst(clamp, Clamp.Min).ToArray<float>();
            var max = _context.GetArgumentConst(clamp, Clamp.Max).ToArray<float>();
            return torch.clamp(input, min, max);
        }
    }
}