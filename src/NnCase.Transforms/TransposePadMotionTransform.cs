using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NnCase.IR;
using NnCase.IR.Operators;

namespace NnCase.Transforms
{
    public class TransposePadMotionTransform : Transform
    {
        protected override bool OnTryMatch(Node node, TransformContext context)
        {
            if (node is Pad pad)
            {
                if (NodeTreeHelper.TryGetDirectParent<Transpose>(pad, out var tp))
                {
                    context.Inputs.Add(tp.Input);
                    context.Outputs.Add(pad.Output);

                    context.MatchedNodes.Add(tp);
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

            var oldTp = (Transpose)context.MatchedNodes[0];
            var oldPad = (Pad)context.MatchedNodes[1];
            var newPad = new Padding[oldPad.Paddings.Count];
            for (int i = 0; i < newPad.Length; i++)
                newPad[i] = oldPad.Paddings[oldTp.Perm[i]];
            /*
            newReduce.Input.Connect(output);

            foreach (var input in inputs.ToList())
                input.Connect(newTp.Output);*/
        }
    }
}
