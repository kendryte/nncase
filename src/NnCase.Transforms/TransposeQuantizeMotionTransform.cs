using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NnCase.IR;
using NnCase.IR.Operators;

namespace NnCase.Transforms
{
    public class TransposeQuantizeMotionTransform : Transform
    {
        protected override bool OnTryMatch(Node node, TransformContext context)
        {
            if (node is Transpose tp)
            {
                if (NodeTreeHelper.TryGetDirectChild<Quantize>(tp, out var q))
                {
                    context.Inputs.Add(tp.Input);
                    context.Outputs.Add(q.Output);

                    context.MatchedNodes.Add(tp);
                    context.MatchedNodes.Add(q);
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
            var oldQ = (Quantize)context.MatchedNodes[1];

            var newQ = context.Graph.AddNode(new Quantize(oldTp.Input.Shape, oldQ.QuantizationParam));
            var newTp = context.Graph.AddNode(new Transpose(newQ.Output.Type, newQ.Output.Shape, oldTp.Perm));
            newTp.Input.Connect(newQ.Output);

            newQ.Input.Connect(output);

            foreach (var input in inputs.ToList())
                input.Connect(newTp.Output);
        }
    }
}
