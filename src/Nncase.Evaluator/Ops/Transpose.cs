using System;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using TorchSharp;

using torchF = TorchSharp.torch.nn.functional;
namespace Nncase.Evaluator.Ops
{
    public sealed partial class EvaluatorVisitor
    {
        private torch.Tensor VisitTranspose(Transpose tr)
        {
            var input = _context.GetArgument(tr, Transpose.Input);
            var perm = _context.GetArgumentConst(tr, Transpose.Perm);
            return input.permute(perm.ToTensor<long>().ToArray());
        }
    }
}