using System;
using Nncase.IR.Tensors;
using TorchSharp;
using Nncase.IR;
using torchF = TorchSharp.torch.nn.functional;
using static Tensorflow.Binding;
using Nncase.IR;
namespace Nncase.Evaluator.Ops
{
    public class ReduceWindow2DEvaluator : IEvaluator<ReduceWindow2D>
    {
        public Const Visit(EvaluatorContext context, ReduceWindow2D r)
        {
            var input = context.GetTorchArgument(r, ReduceWindow2D.Input);
            var kernelSize = context.GetArgumentConstArray<long>(r, ReduceWindow2D.Filter);
            var stride = context.GetArgumentConstArray<long>(r, ReduceWindow2D.Stride);
            var padding = context.GetArgumentConstArray<long>(r, ReduceWindow2D.Padding);
            var countIncludePad = context.GetArgumentConstScalar<bool>(r, ReduceWindow2D.CountIncludePad);
            var ceilMode = context.GetArgumentConstScalar<bool>(r, ReduceWindow2D.CeilMode);
            var afterPad = torchF.pad(input, padding);
            var zeroPadding = new[] {0L, 0};
            return (r.ReduceOp switch
            {
                // avg_pool padding can only pad one side
                ReduceOp.Mean => torchF.avg_pool2d(afterPad, kernelSize, stride, zeroPadding, ceilMode, countIncludePad),
                ReduceOp.Max => torchF.max_pool2d(afterPad, kernelSize, stride, zeroPadding, new[] {1L, 1}, ceilMode),
                _ => throw new ArgumentOutOfRangeException()
            }).ToConst();
        }
    }
}