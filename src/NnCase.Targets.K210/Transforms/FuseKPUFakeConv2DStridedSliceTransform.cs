using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NnCase.IR;
using NnCase.IR.Operators;
using NnCase.Targets.K210.IR;
using NnCase.Targets.K210.IR.FakeOperators;
using NnCase.Transforms;

namespace NnCase.Targets.K210.Transforms
{
    public class FuseKPUFakeConv2DStridedSliceTransform : Transform
    {
        protected override bool OnTryMatch(Node node, TransformContext context)
        {
            if (node is KPUFakeConv2D conv2d && !conv2d.IsDepthwise)
            {
                if (NodeTreeHelper.TryGetDirectChild<StridedSlice>(conv2d, out var slice)
                    && slice.Strides == new Shape(1, 1, 2, 2))
                {
                    context.Inputs.Add(conv2d.Input);
                    context.Outputs.Add(slice.Output);
                    context.MatchedNodes.Add(conv2d);
                    context.MatchedNodes.Add(slice);
                    return true;
                }
            }

            return false;
        }

        public override void Process(TransformContext context)
        {
            var output = context.Inputs[0].Connection;
            var inputs = context.Outputs[0].Connections;
            var oldConv2D = (KPUFakeConv2D)context.MatchedNodes[0];
            var oldSlice = (StridedSlice)context.MatchedNodes[1];

            var padH = new Padding { Before = oldSlice.Begin[2] % 2, After = 0 };
            // pad to even
            if ((oldConv2D.Input.Shape[2] + padH.Before) % 2 == 1)
                padH.After += 1;
            var padW = new Padding { Before = 0, After = 0 };
            var poolType = oldSlice.Begin[3] % 2 == 0 ? KPUPoolType.Pool_LeftTop_2_S2 : KPUPoolType.Pool_RightTop_2_S2;
            var pad = context.Graph.AddNode(new Pad(DataType.Float32, oldConv2D.Input.Shape, new[] { Padding.Zero, Padding.Zero, padH, padW }, 0.0f));
            var conv2d = context.Graph.AddNode(new KPUFakeConv2D(pad.Output.Shape, oldConv2D.IsDepthwise, oldConv2D.FilterType, poolType, oldConv2D.Weights, oldConv2D.Bias, oldConv2D.FusedActivation));
            var cropH = new Padding { Before = -(oldSlice.Begin[2] % 2), After = (oldSlice.End[2] - oldSlice.Input.Shape[2]) / 2 };
            var cropW = new Padding { Before = -(oldSlice.Begin[3] / 2), After = (oldSlice.End[3] - oldSlice.Input.Shape[3]) / 2 };
            var crop = context.Graph.AddNode(new Pad(DataType.Float32, conv2d.Output.Shape, new[] { Padding.Zero, Padding.Zero, cropH, cropW }, 0.0f));
            pad.Input.Connect(output);
            conv2d.Input.Connect(pad.Output);
            crop.Input.Connect(conv2d.Output);

            foreach (var input in inputs.ToList())
                input.Connect(crop.Output);
        }

        private static Padding GetPadding(Padding pad, bool pre)
        {
            if (pre)
                return new Padding { Before = pad.Before > 0 ? pad.Before : 0, After = pad.After > 0 ? pad.After : 0 };
            else
                return new Padding { Before = pad.Before < 0 ? pad.Before : 0, After = pad.After < 0 ? pad.After : 0 };
        }

        private static KPUFilterType GetFilterType(Conv2D conv2D)
        {
            if (conv2D.Weights.Dimensions[2] == 1)
                return KPUFilterType.Filter_1x1;
            else
                return KPUFilterType.Filter_3x3;
        }
    }
}
