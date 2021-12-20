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
        private torch.Tensor VisitExpand(Expand expand)
        {
            var input = _context.GetArgument(expand, Expand.Input);
            var shape = _context.GetArgumentConst(expand, Expand.Shape).ToArray<long>();
            return input.expand(shape);
        }
    }
}