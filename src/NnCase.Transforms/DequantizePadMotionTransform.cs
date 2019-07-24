using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NnCase.IR;
using NnCase.IR.Operators;

namespace NnCase.Transforms
{
    public class DequantizePadMotionTransform : Transform
    {
        protected override bool OnTryMatch(Node node, TransformContext context)
        {
            if (node is Dequantize deq)
            {
                if (NodeTreeHelper.TryGetDirectChild<Pad>(deq, out var pad))
                {
                    context.Inputs.Add(deq.Input);
                    context.Outputs.Add(pad.Output);

                    context.MatchedNodes.Add(deq);
                    context.MatchedNodes.Add(pad);
                    return true;
                }
            }

            return false;
        }

        public override void Process(TransformContext context)
        {
            var output = context.Inputs[0].Connection;
            var inputs = context.Outputs[0].Connections;

            var oldDeq = (Dequantize)context.MatchedNodes[0];
            var oldPad = (Pad)context.MatchedNodes[1];

            var padValue = (byte)Math.Clamp((int)MathF.Round(oldPad.PadValue.As<float>() * oldDeq.QuantizationParam.Scale + oldDeq.QuantizationParam.ZeroPoint), 0, 255);
            var pad = context.Graph.AddNode(new Pad(output.Type, output.Shape, oldPad.Paddings, padValue));
            var deq = context.Graph.AddNode(new Dequantize(pad.Output.Shape, oldDeq.QuantizationParam));
            pad.Input.Connect(output);
            deq.Input.Connect(pad.Output);

            foreach (var input in inputs.ToList())
                input.Connect(deq.Output);
        }
    }
}
