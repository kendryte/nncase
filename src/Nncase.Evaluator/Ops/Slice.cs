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
        private torch.Tensor VisitSlice(Slice sl)
        {
            var input = _context.GetArgument(sl, Slice.Input);
            var begins = _context.GetArgument(sl, Slice.Begins);
            var ends = _context.GetArgument(sl, Slice.Ends);
            var axes = _context.GetArgument(sl, Slice.Axes);
            var strides = _context.GetArgument(sl, Slice.Strides);
            
            var indices = Enumerable.Range(0, input.shape.Length).Select(i =>
                axes.data<int>().Contains(i)? 
                    torch.TensorIndex.Slice(
                        GetItem(begins, i),
                        GetItem(ends, i),
                        GetItem(strides, i)):
                    torch.TensorIndex.Slice()
            ).ToArray();
            return input[indices];
        }

        private long GetItem(torch.Tensor tensor, int index)
        {
            if (tensor.shape.Rank != 1)
            {
                throw new NotSupportedException("Unsupported Rank which > 1 in GetItem(tensor, index)");
            }

            return tensor.to_type(torch.ScalarType.Int64).ReadCpuInt64(index);
        }
    }
}