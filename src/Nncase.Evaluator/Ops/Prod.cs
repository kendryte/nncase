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
    public class ProdEvaluator : IEvaluator<Prod>
    {
        public Const Visit(EvaluatorContext context, Prod prod)
        {
            var input = context.GetTorchArgument(prod, Prod.Input);
            var size = input.shape.Aggregate(1L, (sum, v) => sum * v);
            var v = input.reshape(size).cumprod(0)[size - 1];
            return v.ToConst();
        }
    }
}