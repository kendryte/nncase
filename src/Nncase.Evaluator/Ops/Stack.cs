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
            var axis = _context.GetTorchArgument(stack, Stack.Axis);
            var inputTensors = (inputs as IR.Tuple).Select(x => _context.GetTorchArgument(x)).ToArray();
            return torch.stack(inputTensors, axis.ToScalar().ToInt64());
        }
    }
}