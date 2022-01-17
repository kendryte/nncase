using System;
using Nncase.IR.Tensors;
using Tensorflow;
using static Tensorflow.Binding;
using Nncase.IR;

namespace Nncase.Evaluator.Ops
{
    public class ReverseSequenceEvaluator : IEvaluator<ReverseSequence>
    {
        private Const Visit(EvaluatorContext context, ReverseSequence random)
        {
            var input = context.GetTFArgument(random, ReverseSequence.Input);
            var seqLens = context.GetTFArgument(random, ReverseSequence.SeqLens);
            var batchAxis = context.GetArgumentConstScalar<int>(random, ReverseSequence.BatchAxis);
            var timeAxis = context.GetArgumentConstScalar<int>(random, ReverseSequence.TimeAxis);
            return tf.Context.ExecuteOp("ReverseSequence", null,
                new ExecuteOpArgs(input, seqLens)
                    .SetAttributes(new
                    {
                        seq_dim = timeAxis,
                        batch_dim = batchAxis
                    }))[0].ToConst();
        }
    }
}