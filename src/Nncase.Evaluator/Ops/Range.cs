using System;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using TorchSharp;
using Range = Nncase.IR.Tensors.Range;
using torchF = TorchSharp.torch.nn.functional;
namespace Nncase.Evaluator.Ops
{
    public sealed partial class EvaluatorVisitor
    {
        private torch.Tensor VisitRange(Range range)
        {
            var begin = _context.GetArgumentConstScalar<int>(range, Range.Begin);
            var end = _context.GetArgumentConstScalar<int>(range, Range.End);
            var step = _context.GetArgumentConstScalar<int>(range, Range.Step);
            return torch.arange(begin, end, step);
        }
    }
}