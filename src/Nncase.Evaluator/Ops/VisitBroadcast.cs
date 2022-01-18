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
    public class BroadcastEvaluator : IEvaluator<Broadcast>
    {
        public Const Visit(EvaluatorContext context, Broadcast b)
        {
            var input = context.GetTorchArgument(b, Broadcast.Input);
            var shape = context.GetArgumentConst(b, Broadcast.Shape);
            var s = shape.ToArray<long>();
            return input.broadcast_to(s).ToConst();
        }
    }
}