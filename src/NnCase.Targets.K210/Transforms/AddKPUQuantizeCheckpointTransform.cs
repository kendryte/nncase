using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NnCase.IR;
using NnCase.IR.Operators;
using NnCase.Targets.K210.IR.FakeOperators;
using NnCase.Transforms;

namespace NnCase.Targets.K210.Transforms
{
    public class AddKPUQuantizeCheckpointTransform : Transform
    {
        protected override bool OnTryMatch(Node node, TransformContext context)
        {
            switch (node)
            {
                case KPUFakeConv2D _:
                    if (!NodeTreeHelper.TryGetDirectChild<FakeDequantize>(node, out _))
                    {
                        context.Inputs.Add(node.Inputs[0]);
                        context.Outputs.Add(node.Outputs[0]);
                        context.MatchedNodes.Add(node);
                        return true;
                    }

                    break;
                default:
                    break;
            }

            return false;
        }

        public override void Process(TransformContext context)
        {
            var output = context.Inputs[0].Connection;
            var inputs = context.Outputs[0].Connections.ToList();
            var node = context.MatchedNodes[0];

            var q = context.Graph.AddNode(new FakeQuantize(output.Shape));
            var deq = context.Graph.AddNode(new FakeDequantize(node.Outputs[0].Shape));
            q.Input.Connect(output);
            node.Inputs[0].Connect(q.Output);
            deq.Input.Connect(node.Outputs[0]);

            foreach (var input in inputs)
                input.Connect(deq.Output);
        }
    }
}
