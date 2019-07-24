using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NnCase.IR;
using NnCase.IR.Operators;

namespace NnCase.Transforms
{
    public class FoldNopStridedSliceTransform : Transform
    {
        protected override bool SkipSelfContainedCheck => true;

        protected override bool OnTryMatch(Node node, TransformContext context)
        {
            if (node is StridedSlice slice)
            {
                if (slice.Begin.All(x => x == 0)
                    && slice.End == slice.Input.Shape
                    && slice.Strides.All(x => x == 1))
                {
                    context.Inputs.Add(slice.Input);
                    context.Outputs.Add(slice.Output);
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

            foreach (var input in inputs.ToList())
                input.Connect(output);
        }
    }
}
