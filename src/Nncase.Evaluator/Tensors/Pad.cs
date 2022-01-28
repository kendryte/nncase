
using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.IR.Tensors;
using Tensorflow;
using Tensorflow.NumPy;
using static Tensorflow.Binding;
using Nncase.IR;

using torchF = TorchSharp.torch.nn.functional;

namespace Nncase.Evaluator.Ops
{
    public class PadEvaluator : IEvaluator<Pad>
    {
        public Const Visit(EvaluatorContext context, Pad pad)
        {
            var input = context.GetTFArgument(pad, Pad.Input);
            var pads = context.GetTFArgument(pad, Pad.Pads);
            var constant_values = context.GetArgumentConst(pad, Pad.Value).ToScalar<int>();
            var mode = pad.PadMode switch
            {
                PadMode.Constant => "CONSTANT",
                PadMode.Reflect => "REFLECT",
                PadMode.Symmetric => "SYMMETRIC",
                PadMode.Edge => "EDGE",
                _ => throw new ArgumentOutOfRangeException()
            };
            return tf.Context.ExecuteOp("Pad", null,
                new ExecuteOpArgs(input, pads, mode, constant_values))[0].ToConst();

            // return tf.pad(input, pads, mode: mode, constant_values:constant_values).ToConst();
        }
    }
}