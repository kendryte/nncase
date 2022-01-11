using System;
using Nncase.IR.Tensors;
using TorchSharp;
using torchF = TorchSharp.torch.nn.functional;
using static Tensorflow.Binding;
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
            var countIncludePad = _context.GetArgumentConstScalar<bool>(r, ReduceWindow2D.CountIncludePad);
            var ceilMode = _context.GetArgumentConstScalar<bool>(r, ReduceWindow2D.CeilMode);
            var afterPad = torchF.pad(input, padding);
            var zeroPadding = new[] {0L, 0};
            return r.ReduceOp switch
            {
                // avg_pool padding can only pad one side
                ReduceOp.Mean => torchF.avg_pool2d(afterPad, kernelSize, stride, zeroPadding, ceilMode, countIncludePad),
                ReduceOp.Max => torchF.max_pool2d(afterPad, kernelSize, stride, zeroPadding, new[] {1L, 1}, ceilMode),
                _ => throw new ArgumentOutOfRangeException()
            };
        }
    }
}