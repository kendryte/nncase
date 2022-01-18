using System;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using TorchSharp;
using Nncase.IR;
using Nncase;

using torchF = TorchSharp.torch.nn.functional;
namespace Nncase.Evaluator.Ops
{
    public class ShapeEvaluator : IEvaluator<ShapeOp>
    {
        public Const Visit(EvaluatorContext context, ShapeOp shape)
        {
            var input = context.GetTorchArgument(shape, ShapeOp.Input);
            var dtype = context.CurrentCallResultTensorType().DType.ToTorchType();
            return ((torch.Tensor)input.shape).to_type(dtype).ToConst();
        }
    }
}