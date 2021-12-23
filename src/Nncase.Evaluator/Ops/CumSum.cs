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
        private torch.Tensor VisitCumSum(CumSum cumSum)
        {
            var input = _context.GetArgument(cumSum, CumSum.Input);
            // in onnx, CumSum.Axis is a input tensor with one value
            var dim = _context.GetArgumentConst(cumSum, CumSum.Axis).ToTensor<long>()[0];
            var exclusive = _context.GetArgumentConst(cumSum, CumSum.Exclusive).ToScalar<bool>();
            var reverse = _context.GetArgumentConst(cumSum, CumSum.Reverse).ToScalar<bool>();
            return input.cumsum(dim);
        }
    }
}