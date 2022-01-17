using System;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using TorchSharp;
using Nncase.IR;

using torchF = TorchSharp.torch.nn.functional;
namespace Nncase.Evaluator.Ops
{
    public class SizeEvaluator : IEvaluator<Size>
    {
        public static Const Visit(EvaluatorContext context, Size size)
        {
            var input = context.GetTorchArgument(size, Size.Input);
            var v = (Const)((int)input.numel());
            return v;
        }
    }
}