using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NnCase.IR;
using NnCase.IR.Operators;

namespace NnCase.Transforms
{
    public class FoldNopPadTransform : Transform
    {
        protected override bool SkipSelfContainedCheck => true;

        protected override bool OnTryMatch(Node node, TransformContext context)
        {
            if (node is Pad pad)
            {
                if (pad.Paddings.All(x => x.Before == 0 && x.After == 0))
                {
                    context.Inputs.Add(pad.Input);
                    context.Outputs.Add(pad.Output);
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

            foreach (var input in inputs.ToList())
                input.Connect(output);
        }
    }
}
