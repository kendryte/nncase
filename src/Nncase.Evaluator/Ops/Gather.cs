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
        private torch.Tensor VisitGather(Gather gather)
        {
            var input = _context.GetTorchArgument(gather, Gather.Input);
            var axis = _context.GetArgumentConst(gather, Gather.Axis).ToScalar<int>();
            var index = _context.GetArgumentConst(gather, Gather.Index).ToArray<int>();
            return torch.cat(index.Select(i => input.select(axis, i)).ToList(), 0);
        }
    }
}