using System;
using Nncase.IR.Tensors;
using Tensorflow;
using static Tensorflow.Binding;
using Nncase.IR;

namespace Nncase.Evaluator.Ops
{
    public class RandomEvaluator : IEvaluator<RandomNormal>
    {
        public Const Visit(EvaluatorContext context, RandomNormal random)
        {
            var shape = context.GetArgumentConst(random, RandomNormal.Shape).ToArray<int>();
            var mean = context.GetArgumentConstScalar<float>(random, RandomNormal.Mean);
            var scale = context.GetArgumentConstScalar<float>(random, RandomNormal.Scale);
            var seed = context.GetArgumentConstScalar<int>(random, RandomNormal.Seed);
            return tf.random.normal(shape, mean, stddev: scale, seed: seed).ToConst();
        }
    }
}