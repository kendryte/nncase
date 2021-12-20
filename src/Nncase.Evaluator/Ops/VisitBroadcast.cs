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
        private torch.Tensor VisitBroadcast(Broadcast b)
        {
            var input = _context.GetArgument(b, Broadcast.Input);
            var shape = _context.GetArgumentConst(b, Broadcast.Shape);
            var s = shape.ToArray<long>();
            return input.broadcast_to(s);
        }
    }
}