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
    public class ClampEvaluator : IEvaluator<Clamp>
    {
        public static Const Visit(EvaluatorContext context, Clamp clamp)
        {
            var input = context.GetTorchArgument(clamp, Clamp.Input);
            var min = context.GetArgumentConst(clamp, Clamp.Min).ToArray<float>();
            var max = context.GetArgumentConst(clamp, Clamp.Max).ToArray<float>();
            return torch.clamp(input, min, max).ToConst();
        }
    }
}