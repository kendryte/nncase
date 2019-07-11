using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NnCase.IR;
using NnCase.IR.Operators;
using NnCase.Targets.CPU.IR.Operators;
using NnCase.Transforms;

namespace NnCase.Targets.CPU.Transforms
{
    public class CPUReduceWindow2DTransform : Transform
    {
        protected override bool OnTryMatch(Node node, TransformContext context)
        {
            if (node is ReduceWindow2D reduce2D)
            {
                if (NodeTreeHelper.TryGetDirectParent<Transpose>(reduce2D, out _))
                {
                    context.Inputs.Add(reduce2D.Input);
                    context.Outputs.Add(reduce2D.Output);
                    context.MatchedNodes.Add(reduce2D);
                    return true;
                }
            }

            return false;
        }

        public override void Process(TransformContext context)
        {
            var output = context.Inputs[0].Connection;
            var inputs = context.Outputs[0].Connections;
            var oldReduce2D = (ReduceWindow2D)context.MatchedNodes[0];

            var preTp = context.Graph.NCHWToNHWC(output.Type, output.Shape);
            var reduce2D = context.Graph.AddNode(new CPUReduceWindow2D(oldReduce2D.ReduceOperator, oldReduce2D.InitialValue, preTp.Output.Shape, oldReduce2D.FilterH, oldReduce2D.FilterW, oldReduce2D.PaddingH, oldReduce2D.PaddingW, oldReduce2D.StrideH, oldReduce2D.StrideW, oldReduce2D.DilationH, oldReduce2D.DilationW, oldReduce2D.FusedActivation));
            var surTp = context.Graph.NHWCToNCHW(reduce2D.Output.Type, reduce2D.Output.Shape);
            preTp.Input.Connect(output);
            reduce2D.Input.Connect(preTp.Output);
            surTp.Input.Connect(reduce2D.Output);

            foreach (var input in inputs.ToList())
                input.Connect(surTp.Output);
        }
    }
}
