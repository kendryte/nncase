using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NnCase.IR;
using NnCase.IR.Operators;

namespace NnCase.Transforms
{
    // Transpose (perm = p1)
    //     |
    //     v
    // Transpose (perm = p2)
    //
    // p1[p2[i]] == i

    public class FoldTransposeTransform : Transform
    {
        protected override bool SkipSelfContainedCheck => true;

        protected override bool OnTryMatch(Node node, TransformContext context)
        {
            if (node is Transpose tp1)
            {
                if (NodeTreeHelper.TryGetDirectChild<Transpose>(tp1, out var tp2))
                {
                    if (tp1.Perm.Count == tp2.Perm.Count)
                    {
                        for (int i = 0; i < tp1.Perm.Count; i++)
                        {
                            if (tp1.Perm[tp2.Perm[i]] != i)
                                return false;
                        }

                        context.Inputs.Add(tp1.Input);
                        context.Outputs.Add(tp2.Output);

                        context.MatchedNodes.Add(tp1);
                        context.MatchedNodes.Add(tp2);
                        return true;
                    }
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
