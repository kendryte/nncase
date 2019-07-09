using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NnCase.IR;
using NnCase.IR.Operators;

namespace NnCase.Transforms
{
    public class FoldNopTransposeTransform : Transform
    {
        protected override bool SkipSelfContainedCheck => true;

        protected override bool OnTryMatch(Node node, TransformContext context)
        {
            if (node is Transpose tp)
            {
                for (int i = 0; i < tp.Perm.Count; i++)
                {
                    if (tp.Perm[i] != i)
                        return false;
                }

                context.Inputs.Add(tp.Input);
                context.Outputs.Add(tp.Output);
                context.MatchedNodes.Add(tp);
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
