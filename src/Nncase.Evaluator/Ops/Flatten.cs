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
    public class FlattenEvaluator : IEvaluator<Flatten>
    {
        public static Const Visit(EvaluatorContext context, Flatten flatten)
        {
            var input = context.GetTorchArgument(flatten, Flatten.Input);
            var dim = context.GetArgumentConst(flatten, Flatten.Axis).ToScalar<int>();
            var v = torch.nn.Flatten(0, dim);
            return v.forward(input).ToConst();
        }
    }
}