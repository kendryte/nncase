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
        private torch.Tensor VisitExpand(Expand expand)
        {
            var input = _context.GetTorchArgument(expand, Expand.Input);
            if (input.shape.Length == 0)
            {
                input = input.reshape(1L);
            }
            var shape = _context.GetArgumentConst(expand, Expand.Shape).ToArray<long>();
            // When the value of onnx is 1, the value of torch is -1
            var torchShape = shape.Select(x => x == 1 ? -1 : x).ToArray();
            if (torchShape.Length < input.shape.Length)
            {
                // [-1]*n.Concat(TorchShape)
                torchShape = Enumerable.Repeat(-1L, input.shape.Length - torchShape.Length).Concat(torchShape).ToArray();
            }
            return input.expand(torchShape);
        }
    }
}