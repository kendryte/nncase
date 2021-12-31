using System;
using Nncase.IR.Tensors;
using TorchSharp;
using torchF = TorchSharp.torch.nn.functional;

namespace Nncase.Evaluator.Ops
{
    public sealed partial class EvaluatorVisitor
    {
        private torch.Tensor VisitReduceWindow2D(ReduceWindow2D r)
        {
            var input = _context.GetTorchArgument(r, ReduceWindow2D.Input);
            var kernelSize = _context.GetArgumentConstArray<long>(r, ReduceWindow2D.Filter);
            var stride = _context.GetArgumentConstArray<long>(r, ReduceWindow2D.Stride);
            var padding = _context.GetArgumentConstArray<long>(r, ReduceWindow2D.Padding);
            var ceilMode = _context.GetArgumentConstScalar<bool>(r, ReduceWindow2D.CeilMode);
            var afterPad = torchF.pad(input, padding);
            var zeroPadding = new[] {0L, 0};
            return r.ReduceOp switch
            {
                ReduceOp.Mean => torchF.avg_pool2d(afterPad, kernelSize, stride, zeroPadding, ceilMode),
                ReduceOp.Min => -torchF.max_pool2d(-afterPad, kernelSize, stride, zeroPadding, new[] {1L, 1}, ceilMode),
                ReduceOp.Max => torchF.max_pool2d(afterPad, kernelSize, stride, zeroPadding, new[] {1L, 1}, ceilMode),
                _ => throw new ArgumentOutOfRangeException()
            };
        }
    }
}