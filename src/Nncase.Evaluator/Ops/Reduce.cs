using System;
using System.Linq;
using System.Collections.Generic;
using NetFabric.Hyperlinq;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using TorchSharp;

namespace Nncase.Evaluator.Ops
{
    public sealed partial class EvaluatorVisitor
    {
        private torch.Tensor VisitReduce(Reduce reduce)
        {
            var input = _context.GetTorchArgument(reduce, Reduce.Input);
            var dims = _context.GetArgumentConstArray<long>(reduce, Reduce.Axis);
            var keepDims = _context.GetArgumentConstScalar<bool>(reduce, Reduce.KeepDims);
            var initValue = _context.GetArgumentConstScalar<float>(reduce, Reduce.InitValue);

            return reduce.ReduceOp switch
            {
                ReduceOp.Mean => torch.mean(input, dims, keepDims),
                // ReduceOp.Min => torch.min(input, dims, keepDims),
                // ReduceOp.Max => torch.max(input, dims, keepDims),
                ReduceOp.Sum => input.sum(dims, keepDims),
                _ => throw new ArgumentOutOfRangeException(),
            };
        }
    }
}