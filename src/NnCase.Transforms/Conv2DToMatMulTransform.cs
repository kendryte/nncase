using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NnCase.IR;
using NnCase.IR.Operators;

namespace NnCase.Transforms
{
    public class Conv2DToMatMulTransform : Transform
    {
        protected override bool SkipSelfContainedCheck => true;

        protected override bool OnTryMatch(Node node, TransformContext context)
        {
            if (node is Conv2D conv2d
                && conv2d.Input.Shape[1] == 1 && conv2d.Input.Shape[2] == 1
                && conv2d.PaddingH == Padding.Zero && conv2d.PaddingW == Padding.Zero
                && conv2d.StrideH == 1 && conv2d.StrideW == 1)
            {
                context.Inputs.Add(conv2d.Input);
                context.Outputs.Add(conv2d.Output);
                context.MatchedNodes.Add(conv2d);
                return true;
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
