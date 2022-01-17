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
    public class TransposeEvaluator : IEvaluator<Transpose>
    {
        public static Const Visit(EvaluatorContext context, Transpose tr)
        {
            var input = context.GetTorchArgument(tr, Transpose.Input);
            var perm = context.GetArgumentConst(tr, Transpose.Perm);
            return input.permute(perm.ToTensor<long>().ToArray()).ToConst();
        }
    }
}