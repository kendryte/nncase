using System;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using TorchSharp;
using Nncase;

using torchF = TorchSharp.torch.nn.functional;
namespace Nncase.Evaluator.Ops
{
    public sealed partial class EvaluatorVisitor
    {
        private torch.Tensor VisitShape(ShapeOp shape)
        {
            var input = _context.GetArgument(shape, ShapeOp.Input);
            var dtype = _context.CurrentCallResultTensorType().DType.ToTorchType();
            return ((torch.Tensor)input.shape).to_type(dtype);
        }
    }
}