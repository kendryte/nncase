using System;
using Nncase.IR.Tensors;
using Tensorflow;
using static Tensorflow.Binding;

namespace Nncase.Evaluator.Ops
{
    public sealed partial class EvaluatorVisitor
    {
        private Tensor VisitReverseSequence(ReverseSequence random)
        {
            var input = _context.GetTFArgument(random, ReverseSequence.Input);
            var seqLens = _context.GetTFArgument(random, ReverseSequence.SeqLens);
            var batchAxis = _context.GetArgumentConstScalar<int>(random, ReverseSequence.BatchAxis);
            var timeAxis = _context.GetArgumentConstScalar<int>(random, ReverseSequence.TimeAxis);
            return tf.Context.ExecuteOp("ReverseSequence", null,
                new ExecuteOpArgs(input, seqLens)
                    .SetAttributes(new
                    {
                        seq_dim = timeAxis,
                        batch_dim = batchAxis
                    }));
        }
    }
}