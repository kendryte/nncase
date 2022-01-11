using System;
using Nncase.IR.Tensors;
using Tensorflow;
using static Tensorflow.Binding;

namespace Nncase.Evaluator.Ops
{
    public sealed partial class EvaluatorVisitor
    {
        private Tensor VisitRandomNormal(RandomNormal random)
        {
            var shape = _context.GetArgumentConst(random, RandomNormal.Shape).ToArray<int>();
            var mean = _context.GetArgumentConstScalar<float>(random, RandomNormal.Mean);
            var scale = _context.GetArgumentConstScalar<float>(random, RandomNormal.Scale);
            var seed = _context.GetArgumentConstScalar<int>(random, RandomNormal.Seed);
            return tf.random.normal(shape, mean, stddev:scale, seed: seed);
        }
    }
}