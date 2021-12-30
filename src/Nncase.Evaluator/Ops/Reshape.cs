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
        private torch.Tensor VisitReshape(Reshape reshape)
        {
            var input = _context.GetTorchArgument(reshape, Reshape.Input);
            var shape = _context.GetArgumentConst(reshape, Reshape.Shape).ToArray<long>();
            return input.reshape(shape);
        }
    }
}