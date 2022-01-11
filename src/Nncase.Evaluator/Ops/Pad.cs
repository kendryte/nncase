
using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.IR.Tensors;
using Tensorflow;
using Tensorflow.NumPy;
using static Tensorflow.Binding;

using torchF = TorchSharp.torch.nn.functional;

namespace Nncase.Evaluator.Ops
{
    public sealed partial class EvaluatorVisitor
    {
        private Tensorflow.Tensor VisitPad(Pad pad)
        {
            var input = _context.GetTFArgument(pad, Pad.Input);
            var pads = _context.GetTFArgument(pad, Pad.Pads);
            var constant_values = _context.GetArgumentConst(pad, Pad.Value).ToScalar<int>();
            var mode = pad.PadMode switch
            {
                PadMode.Constant => "CONSTANT",
                PadMode.Reflect => "REFLECT",
                PadMode.Symmetric => "SYMMETRIC",
                PadMode.Edge => "EDGE",
                _ => throw new ArgumentOutOfRangeException()
            };
            return tf.Context.ExecuteOp("Pad", null,
                new ExecuteOpArgs(input, pads, mode, constant_values));
            // return tf.pad(input, pads, mode: mode, constant_values:constant_values);
        }
    }
}