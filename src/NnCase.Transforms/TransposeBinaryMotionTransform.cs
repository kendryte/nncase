using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NnCase.IR;
using NnCase.IR.Operators;

namespace NnCase.Transforms
{
    public class TransposeBinaryMotionTransform : Transform
    {
        protected override bool OnTryMatch(Node node, TransformContext context)
        {
            if (node is Binary binary)
            {
                if (binary.InputA.Connection.Owner is Transpose tp1 &&
                    binary.InputB.Connection.Owner is Transpose tp2)
                {
                    if (tp1.Perm == tp2.Perm)
                    {
                        context.Inputs.Add(tp1.Input);
                        context.Inputs.Add(tp2.Input);
                        context.Outputs.Add(binary.Output);

                        context.MatchedNodes.Add(tp1);
                        context.MatchedNodes.Add(tp2);
                        context.MatchedNodes.Add(binary);
                        return true;
                    }
                }
            }

            return false;
        }

        public override void Process(TransformContext context)
        {
            var outputA = context.Inputs[0].Connection;
            var outputB = context.Inputs[1].Connection;
            var inputs = context.Outputs[0].Connections;

            var oldTp = (Transpose)context.MatchedNodes[0];
            var oldBinary = (Binary)context.MatchedNodes[2];

            var newBinary = context.Graph.AddNode(new Binary(oldBinary.BinaryOperator, outputA.Shape, outputB.Shape, oldBinary.FusedActivation));
            var newTp = context.Graph.AddNode(new Transpose(newBinary.Output.Type, newBinary.Output.Shape, oldTp.Perm));
            newTp.Input.Connect(newBinary.Output);

            newBinary.InputA.Connect(outputA);
            newBinary.InputB.Connect(outputB);

            foreach (var input in inputs.ToList())
                input.Connect(newTp.Output);
        }
    }
}
