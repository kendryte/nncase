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
    public class ConcatEvaluator : IEvaluator<Concat>
    {
        public static Const Visit(EvaluatorContext context, Concat cat)
        {
            var inputs = context.GetArgumentExpr(cat, Concat.Input);
            var axis = context.GetArgumentConst(cat, Concat.Axis).ToScalar<int>();
            var inputTensors = (inputs as IR.Tuple).Select(x => expandDim(context.GetTorchArgument(x))).ToArray();
            return torch.cat(inputTensors, axis).ToConst();
        }

        internal static torch.Tensor expandDim(torch.Tensor tensor)
        {
            if (!tensor.shape.Any())
                return tensor.view(new long[] { 1 });
            return tensor;
        }
    }
}