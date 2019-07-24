using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NnCase.IR;
using NnCase.IR.Operators;

namespace NnCase.Transforms
{
    public class PadQuantizeMotionTransform : Transform
    {
        protected override bool OnTryMatch(Node node, TransformContext context)
        {
            if (node is Pad pad)
            {
                if (NodeTreeHelper.TryGetDirectChild<Quantize>(pad, out var q))
                {
                    context.Inputs.Add(pad.Input);
                    context.Outputs.Add(q.Output);

                    context.MatchedNodes.Add(pad);
                    context.MatchedNodes.Add(q);
                    return true;
                }
            }

            return false;
        }

        public override void Process(TransformContext context)
        {
            var output = context.Inputs[0].Connection;
            var inputs = context.Outputs[0].Connections;

            var oldPad = (Pad)context.MatchedNodes[0];
            var oldQ = (Quantize)context.MatchedNodes[1];

            var padValue = (byte)Math.Clamp((int)MathF.Round(oldPad.PadValue.As<float>() * oldQ.QuantizationParam.Scale + oldQ.QuantizationParam.ZeroPoint), 0, 255);
            var q = context.Graph.AddNode(new Quantize(output.Shape, oldQ.QuantizationParam));
            var pad = context.Graph.AddNode(new Pad(q.Output.Type, q.Output.Shape, oldPad.Paddings, padValue));
            q.Input.Connect(output);
            pad.Input.Connect(q.Output);

            foreach (var input in inputs.ToList())
                input.Connect(pad.Output);
        }
    }
}
