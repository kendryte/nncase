using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NnCase.IR;
using NnCase.IR.Operators;

namespace NnCase.Transforms
{
    public class TransposeReduceMotionTransform : Transform
    {
        protected override bool OnTryMatch(Node node, TransformContext context)
        {
            if (node is Reduce reduce)
            {
                if (NodeTreeHelper.TryGetDirectParent<Transpose>(reduce, out var tp))
                {
                    context.Inputs.Add(tp.Input);
                    context.Outputs.Add(reduce.Output);

                    context.MatchedNodes.Add(tp);
                    context.MatchedNodes.Add(reduce);
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
            var oldReduce = (Reduce)context.MatchedNodes[1];
            var newAxis = oldReduce.Axis.ToArray().Select(x => oldTp.Perm[x]).ToList();
            var newPerm = oldReduce.KeepDims
                ? oldTp.Perm
                : new Shape(oldTp.Perm.ToArray().Where(x => !oldReduce.Axis.Contains(x)).Select(x => x - oldReduce.Axis.CountIf(a => a < x)).ToList());

            var newReduce = context.Graph.AddNode(new Reduce(oldReduce.ReduceOperator, oldReduce.InitialValue, output.Shape, new Shape(newAxis), oldReduce.KeepDims));
            var newTp = context.Graph.AddNode(new Transpose(newReduce.Output.Type, newReduce.Output.Shape, newPerm));
            newTp.Input.Connect(newReduce.Output);

            newReduce.Input.Connect(output);

            foreach (var input in inputs.ToList())
                input.Connect(newTp.Output);
        }
    }
}
