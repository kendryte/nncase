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
        private torch.Tensor VisitCast(Cast cast)
        {
            var input = _context.GetTorchArgument(cast, Cast.Input);
            return input.to_type(cast.NewType.ToTorchType());
        }
    }
}