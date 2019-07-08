using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NnCase.IR;
using NnCase.IR.Operators;

namespace NnCase.Transforms
{
    public class TransposeConcatMotionTransform : Transform
    {
        protected override bool OnTryMatch(Node node, TransformContext context)
        {
            if (node is Concat concat)
            {
                context.Outputs.Add(concat.Output);
                context.MatchedNodes.Add(concat);

                Shape perm = null;
                foreach (var input in concat.Inputs)
                {
                    if (input.Connection.Owner is Transpose tp)
                    {
                        if (perm == null)
                            perm = tp.Perm;
                        else if (perm != tp.Perm)
                            return false;

                        context.Inputs.Add(tp.Input);
                        context.MatchedNodes.Add(tp);
                    }
                    else
                    {
                        return false;
                    }
                }

                return true;
            }

            return false;
        }

        public override void Process(TransformContext context)
        {
            var outputs = context.Inputs.Select(x => x.Connection);
            var inputs = context.Outputs[0].Connections;
            var perm = ((Transpose)context.MatchedNodes[1]).Perm;

            var oldConcat = (Concat)context.MatchedNodes[0];

            var newAxis = perm[oldConcat.Axis];
            var newConcat = context.Graph.AddNode(new Concat(oldConcat.Output.Type, outputs.Select(x => x.Shape), newAxis));
            var newTp = context.Graph.AddNode(new Transpose(newConcat.Output.Type, newConcat.Output.Shape, perm));
            newTp.Input.Connect(newConcat.Output);

            for (int i = 0; i < newConcat.Inputs.Count; i++)
                newConcat.Inputs[i].Connect(context.Inputs[i].Connection);

            foreach (var input in inputs.ToList())
                input.Connect(newTp.Output);
        }
    }
}
