using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NnCase.IR;
using NnCase.IR.Operators;

namespace NnCase.Transforms
{
    public class DequantizeStridedSliceMotionTransform : Transform
    {
        protected override bool OnTryMatch(Node node, TransformContext context)
        {
            if (node is Dequantize deq)
            {
                if (NodeTreeHelper.TryGetDirectChild<StridedSlice>(deq, out var slice))
                {
                    context.Inputs.Add(deq.Input);
                    context.Outputs.Add(slice.Output);

                    context.MatchedNodes.Add(deq);
                    context.MatchedNodes.Add(slice);
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
            var oldSlice = (StridedSlice)context.MatchedNodes[1];

            var slice = context.Graph.AddNode(new StridedSlice(output.Type, output.Shape, oldSlice.Begin, oldSlice.End, oldSlice.Strides, oldSlice.BeginMask, oldSlice.EndMask, oldSlice.EllipsisMask, oldSlice.NewAxisMask, oldSlice.ShrinkAxisMask));
            var deq = context.Graph.AddNode(new Dequantize(slice.Output.Shape, oldDeq.QuantizationParam));
            slice.Input.Connect(output);
            deq.Input.Connect(slice.Output);

            foreach (var input in inputs.ToList())
                input.Connect(deq.Output);
        }
    }
}
