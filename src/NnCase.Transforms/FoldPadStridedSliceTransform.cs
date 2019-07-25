using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NnCase.IR;
using NnCase.IR.Operators;

namespace NnCase.Transforms
{
    public class FoldPadStridedSliceTransform : Transform
    {
        protected override bool OnTryMatch(Node node, TransformContext context)
        {
            if (node is Pad pad
                && pad.Paddings.Any(x => x.Before < 0 || x.After < 0))
            {
                if (NodeTreeHelper.TryGetDirectChild<StridedSlice>(pad, out var slice)
                    && slice.Begin.All(x => x >= 0) && slice.End.All(x => x >= 0)
                    && slice.NewAxisMask == 0 && slice.ShrinkAxisMask == 0 && slice.EllipsisMask == 0)
                {
                    context.Inputs.Add(pad.Input);
                    context.Outputs.Add(slice.Output);

                    context.MatchedNodes.Add(pad);
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
            var oldPad = (Pad)context.MatchedNodes[0];
            var oldSlice = (StridedSlice)context.MatchedNodes[1];

            var paddings = oldPad.Paddings.ToArray();
            var begin = oldSlice.Begin.Clone();
            var end = oldSlice.End.Clone();
            for (int i = 0; i < paddings.Length; i++)
            {
                ref var padding = ref paddings[i];
                if (padding.Before < 0)
                {
                    var padBefore = -padding.Before;
                    begin[i] += padBefore;
                    end[i] += padBefore;
                    padding.Before = 0;
                }
                if (padding.After < 0)
                {
                    end[i] += padding.After;
                    padding.After = 0;
                }
            }

            var pad = context.Graph.AddNode(new Pad(output.Type, output.Shape, paddings, oldPad.PadValue));
            var slice = context.Graph.AddNode(new StridedSlice(pad.Output.Type, pad.Output.Shape, begin, end, oldSlice.Strides, oldSlice.BeginMask, oldSlice.EndMask, oldSlice.EllipsisMask, oldSlice.NewAxisMask, oldSlice.ShrinkAxisMask));
            pad.Input.Connect(output);
            slice.Input.Connect(pad.Output);

            foreach (var input in inputs.ToList())
                input.Connect(slice.Output);
        }
    }
}
