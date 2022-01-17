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
    public class ReshapeEvaluator : IEvaluator<Reshape>
    {
        public static Const Visit(EvaluatorContext context, Reshape reshape)
        {
            var input = context.GetTorchArgument(reshape, Reshape.Input);
            var shape = context.GetArgumentConst(reshape, Reshape.Shape).ToArray<long>();
            return input.reshape(shape).ToConst();
        }
    }
}