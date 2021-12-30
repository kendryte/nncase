using System;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using TorchSharp;

using torchF = TorchSharp.torch.nn.functional;
namespace Nncase.Evaluator.Ops
{
    public sealed partial class EvaluatorVisitor
    {
        private torch.Tensor VisitProd(Prod prod)
        {
            var input = _context.GetTorchArgument(prod, Prod.Input);
            var size = input.shape.Aggregate(1L, (sum, v) => sum * v);
            var v = input.reshape(size).cumprod(0)[size - 1];
            return v;
        }
    }
}