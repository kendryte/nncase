using System;
using System.Linq;
using System.Collections.Generic;
using NetFabric.Hyperlinq;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using TorchSharp;

using torchF = TorchSharp.torch.nn.functional;

namespace Nncase.Evaluator.Ops
{
    public sealed partial class EvaluatorVisitor
    {
        private torch.Tensor VisitStack(Stack stack)
        {
            var inputs = _context.GetArgumentExpr(stack, Stack.Inputs);
            var axis = _context.GetArgument(stack, Stack.Axis);
            var inputTensors = ((IR.Tuple)inputs).Select(x => _context.GetArgument(x)).ToArray();
            return torch.stack(inputTensors, axis.ToScalar().ToInt64());
        }
    }
}