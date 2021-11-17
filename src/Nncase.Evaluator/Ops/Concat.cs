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
        private torch.Tensor VisitConcat(Concat cat)
        {
            var inputs = _context.GetArgumentExpr(cat, Concat.Input);
            var axis = _context.GetArgumentConst(cat, Concat.Axis).ToScalar<int>();
            var inputTensors = (inputs as IR.Tuple).Select(x => _context.GetArgument(x)).ToArray();
            return torch.cat(inputTensors, axis);
        }
    }
}