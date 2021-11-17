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
            return input[begins, ends, axes, strides];
        }
    }
}