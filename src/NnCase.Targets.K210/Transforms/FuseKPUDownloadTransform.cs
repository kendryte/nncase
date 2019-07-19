using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NnCase.IR;
using NnCase.IR.Operators;
using NnCase.Targets.K210.IR.FakeOperators;

namespace NnCase.Transforms
{
    public class FuseKPUDownloadTransform : Transform
    {
        protected override bool SkipSelfContainedCheck => true;

        protected override bool OnTryMatch(Node node, TransformContext context)
        {
            if (node is KPUDownload down)
            {
                if (NodeTreeHelper.TryGetDirectParent<KPUConv2D>(down, out var conv2d))
                {
                    context.Inputs.Add(conv2d.Input);
                    context.Outputs.Add(down.Output);

                    context.MatchedNodes.Add(conv2d);
                    context.MatchedNodes.Add(down);
                    return true;
                }
            }

            return false;
        }

        public override void Process(TransformContext context)
        {
            var output = context.Inputs[0].Connection;
            var inputs = context.Outputs[0].Connections;

            var oldConv2d = (KPUConv2D)context.MatchedNodes[0];
            var conv2d = context.Graph.AddNode(new KPUConv2D(oldConv2d.Input.Shape, oldConv2d.IsDepthwise, oldConv2d.FilterType, oldConv2d.PoolType, oldConv2d.Weights, oldConv2d.PadValue, oldConv2d.ArgX, oldConv2d.ShiftX, oldConv2d.ArgW, oldConv2d.ShiftW, oldConv2d.ArgAdd, oldConv2d.BatchNorm, oldConv2d.Activation, true));
            conv2d.Input.Connect(output);

            foreach (var input in inputs.ToList())
                input.Connect(conv2d.MainMemoryOutput);
        }
    }
}
