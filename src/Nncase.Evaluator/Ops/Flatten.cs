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
        private torch.Tensor VisitFlatten(Flatten flatten)
        {
            var input = _context.GetArgument(flatten, Flatten.Input);
            var dim = _context.GetArgumentConst(flatten, Flatten.Axis).ToScalar<int>();
            var v = torch.nn.Flatten(0, dim);
            return v.forward(input);
        }
    }
}