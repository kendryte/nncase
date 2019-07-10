using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NnCase.IR;
using NnCase.IR.Operators;

namespace NnCase.Transforms
{
    public class AddQuantizeCheckpointForConv2DTransform : Transform
    {
        protected override bool SkipSelfContainedCheck => true;

        protected override bool OnTryMatch(Node node, TransformContext context)
        {
            if (node is Conv2D conv2d)
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
