using System;
using System.Linq;
using System.Collections.Generic;
using NetFabric.Hyperlinq;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using static Tensorflow.Binding;

namespace Nncase.Evaluator.Ops
{
    public sealed partial class EvaluatorVisitor
    {
        private Tensorflow.Tensor VisitReduce(Reduce reduce)
        {
            var input = _context.GetTFArgument(reduce, Reduce.Input);
            var axis = _context.GetArgumentConstArray<long>(reduce, Reduce.Axis);
            var keepDims = _context.GetArgumentConstScalar<bool>(reduce, Reduce.KeepDims);

            return reduce.ReduceOp switch
            {
                ReduceOp.Mean => tf.reduce_mean(input, axis, keepDims),
                ReduceOp.Max => tf.reduce_max(input, axis, keepDims),
                ReduceOp.Min => tf.reduce_min(input, axis, keepDims),
                ReduceOp.Prod => tf.reduce_prod(input, axis, keepDims),
                ReduceOp.Sum => tf.reduce_sum(input, axis, keepdims: keepDims),
                _ => throw new ArgumentOutOfRangeException(),
            };
        }
    }
}