
using System;
using System.Collections.Generic;
using Nncase.IR.Tensors;
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
            var value = _context.GetArgumentConst(pad, Pad.Value).ToScalar<int>();
            var mod = pad.PadMode switch
            {
                PadMode.Constant => "CONSTANT",
                PadMode.Reflect => "REFLECT",
                PadMode.Symmetric => "SYMMETRIC",
                PadMode.Edge => "EDGE",
                _ => throw new ArgumentOutOfRangeException()
            };
            return tf.pad(input, pads, mod, constant_values:value);
        }
    }
}