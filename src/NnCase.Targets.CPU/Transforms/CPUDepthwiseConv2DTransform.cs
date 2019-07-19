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
    public class CPUDepthwiseConv2DTransform : Transform
    {
        protected override bool OnTryMatch(Node node, TransformContext context)
        {
            if (node is Conv2D conv2d)
            {
                if (conv2d.Groups == conv2d.Input.Shape[1])
                {
                    //if (NodeTreeHelper.TryGetDirectParent<Transpose>(conv2d, out _))
                    {
                        context.Inputs.Add(conv2d.Input);
                        context.Outputs.Add(conv2d.Output);
                        context.MatchedNodes.Add(conv2d);
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
            var oldConv2D = (Conv2D)context.MatchedNodes[0];

            var newWeights = oldConv2D.Weights.Transpose(new[] { 0, 2, 3, 1 });
            var preTp = context.Graph.NCHWToNHWC(output.Type, output.Shape);
            var conv2d = context.Graph.AddNode(new CPUDepthwiseConv2D(preTp.Output.Shape, newWeights, oldConv2D.Bias, oldConv2D.PaddingH, oldConv2D.PaddingW, oldConv2D.StrideH, oldConv2D.StrideW, oldConv2D.DilationH, oldConv2D.DilationW, oldConv2D.FusedActivation));
            var surTp = context.Graph.NHWCToNCHW(conv2d.Output.Type, conv2d.Output.Shape);
            preTp.Input.Connect(output);
            conv2d.Input.Connect(preTp.Output);
            surTp.Input.Connect(conv2d.Output);

            foreach (var input in inputs.ToList())
                input.Connect(surTp.Output);
        }
    }
}
