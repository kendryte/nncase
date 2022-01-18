using System;
using System.Linq;
using System.Collections.Generic;
using NetFabric.Hyperlinq;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using static Tensorflow.Binding;
using Nncase.IR;

namespace Nncase.Evaluator.Ops
{
    public class ReduceEvaluator : IEvaluator<Reduce>
    {
        public Const Visit(EvaluatorContext context, Reduce reduce)
        {
            var input = context.GetTFArgument(reduce, Reduce.Input);
            var axis = context.GetArgumentConstArray<long>(reduce, Reduce.Axis);
            var keepDims = context.GetArgumentConstScalar<bool>(reduce, Reduce.KeepDims);

            return (reduce.ReduceOp switch
            {
                ReduceOp.Mean => tf.reduce_mean(input, axis, keepDims),
                ReduceOp.Max => tf.reduce_max(input, axis, keepDims),
                ReduceOp.Min => tf.reduce_min(input, axis, keepDims),
                ReduceOp.Prod => tf.reduce_prod(input, axis, keepDims),
                ReduceOp.Sum => tf.reduce_sum(input, axis, keepdims: keepDims),
                _ => throw new ArgumentOutOfRangeException(),
            }).ToConst();
        }
    }
}