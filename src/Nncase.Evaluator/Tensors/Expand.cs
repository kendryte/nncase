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
    public class ExpandEvaluator : IEvaluator<Expand>
    {
        public Const Visit(EvaluatorContext context, Expand expand)
        {
            var input = context.GetTorchArgument(expand, Expand.Input);
            if (input.shape.Length == 0)
            {
                input = input.reshape(1L);
            }
            var shape = context.GetArgumentConst(expand, Expand.Shape).ToArray<long>();

            // When the value of onnx is 1, the value of torch is -1
            var torchShape = shape.Select(x => x == 1 ? -1 : x).ToArray();
            if (torchShape.Length < input.shape.Length)
            {
                // [-1]*n.Concat(TorchShape)
                torchShape = Enumerable.Repeat(-1L, input.shape.Length - torchShape.Length).Concat(torchShape).ToArray();
            }
            return input.expand(torchShape).ToConst();
        }
    }
}