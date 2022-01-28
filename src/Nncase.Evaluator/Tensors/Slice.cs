using System;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using TorchSharp;
using Nncase.IR;

using torchF = TorchSharp.torch.nn.functional;
namespace Nncase.Evaluator.Ops
{
    public class SliceEvaluator : IEvaluator<Slice>
    {
        public Const Visit(EvaluatorContext context, Slice sl)
        {
            var input = context.GetTorchArgument(sl, Slice.Input);
            var begins = context.GetTorchArgument(sl, Slice.Begins);
            var ends = context.GetTorchArgument(sl, Slice.Ends);
            var axes = context.GetArgumentConstArray<int>(sl, Slice.Axes)
                .Select(x => x < 0 ? x + input.shape.Rank : x);
            var strides = context.GetTorchArgument(sl, Slice.Strides);
            var axesIndex = 0;
            var indices = Enumerable.Range(0, input.shape.Length).Select(i =>
                axes.Contains(i) ?
                    torch.TensorIndex.Slice(
                        GetItem(begins, axesIndex),
                        GetItem(ends, axesIndex),
                        GetItem(strides, axesIndex++)) :
                    torch.TensorIndex.Slice()
            ).ToArray();
            return input[indices].ToConst();
        }

        internal static long GetItem(torch.Tensor tensor, int index)
        {
            if (tensor.shape.Rank != 1)
            {
                throw new NotSupportedException("Unsupported Rank which > 1 in GetItem(tensor, index)");
            }

            return tensor.to_type(torch.ScalarType.Int64).ReadCpuInt64(index);
        }
    }
}