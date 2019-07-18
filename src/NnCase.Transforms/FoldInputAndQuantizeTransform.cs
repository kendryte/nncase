using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NnCase.IR;
using NnCase.IR.Operators;

namespace NnCase.Transforms
{
    public class FoldInputAndQuantizeTransform : Transform
    {
        protected override bool SkipSelfContainedCheck => true;

        protected override bool OnTryMatch(Node node, TransformContext context)
        {
            if (node is InputNode inode)
            {
                if (NodeTreeHelper.TryGetDirectChild<Quantize>(inode, out var q))
                {
                    context.Outputs.Add(q.Output);

                    context.MatchedNodes.Add(inode);
                    context.MatchedNodes.Add(q);
                    return true;
                }
            }

            return false;
        }

        public override void Process(TransformContext context)
        {
            var inputs = context.Outputs[0].Connections;
            var oldQ = (Quantize)context.MatchedNodes[1];

            var newInput = context.Graph.AddNode(new InputNode(oldQ.Output.Type, oldQ.Output.Shape, oldQ.Output.MemoryType));

            foreach (var input in inputs.ToList())
                input.Connect(newInput.Output);
        }
    }
}
