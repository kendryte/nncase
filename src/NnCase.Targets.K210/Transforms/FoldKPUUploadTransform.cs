using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NnCase.IR;
using NnCase.IR.Operators;
using NnCase.Targets.K210.IR.FakeOperators;

namespace NnCase.Transforms
{
    public class FoldKPUUploadTransform : Transform
    {
        protected override bool SkipSelfContainedCheck => true;

        protected override bool OnTryMatch(Node node, TransformContext context)
        {
            if (node is KPUUpload up)
            {
                if (NodeTreeHelper.TryGetDirectChild<KPUDownload>(up, out var down))
                {
                    context.Inputs.Add(up.Input);
                    context.Outputs.Add(down.Output);

                    context.MatchedNodes.Add(up);
                    context.MatchedNodes.Add(down);
                    return true;
                }
            }
            else if (node is KPUDownload down)
            {
                if (NodeTreeHelper.TryGetDirectChild<KPUUpload>(down, out up))
                {
                    context.Inputs.Add(down.Input);
                    context.Outputs.Add(up.Output);

                    context.MatchedNodes.Add(down);
                    context.MatchedNodes.Add(up);
                    return true;
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
