using System;
using System.Linq;
using System.Collections.Generic;
using NetFabric.Hyperlinq;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using TorchSharp;
using Nncase.IR;

using torchF = TorchSharp.torch.nn.functional;

namespace Nncase.Evaluator.Ops
{
    public class StackEvaluator : IEvaluator<Stack>
    {
        public Const Visit(EvaluatorContext context, Stack stack)
        {
            var inputs = context.GetArgumentExpr(stack, Stack.Inputs);
            var axis = context.GetTorchArgument(stack, Stack.Axis);
            var inputTensors = (inputs as IR.Tuple).Select(x => context.GetTorchArgument(x)).ToArray();
            return torch.stack(inputTensors, axis.ToScalar().ToInt64()).ToConst();
        }
    }
}