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
    public class KPUFakeConv2DTransform : Transform
    {
        protected override bool OnTryMatch(Node node, TransformContext context)
        {
            if (node is Conv2D conv2d)
            {
                if ((conv2d.Groups == 1 || conv2d.Groups == conv2d.Input.Shape[1])
                    && conv2d.DilationH == 1 && conv2d.DilationW == 1 /* To Be Removed */
                    && ((conv2d.Weights.Dimensions[2] == 1 && conv2d.Weights.Dimensions[3] == 1)
                    || (conv2d.Weights.Dimensions[2] == 3 && conv2d.Weights.Dimensions[3] == 3))
                    && KPUShapeUtility.IsSupportedShape(conv2d.Input.Shape))
                {
                    context.Inputs.Add(conv2d.Input);
                    context.Outputs.Add(conv2d.Output);
                    context.MatchedNodes.Add(conv2d);
                    return true;
                }
            }

            return false;
        }

        public override void Process(TransformContext context)
        {
            var output = context.Inputs[0].Connection;
            var inputs = context.Outputs[0].Connections;
            var oldConv2D = (Conv2D)context.MatchedNodes[0];

            var filterType = GetFilterType(oldConv2D);
            var kpuPad = KPUShapeUtility.GetKPUPadding(filterType);
            var padH = new Padding { Before = oldConv2D.PaddingH.Before - kpuPad, After = oldConv2D.PaddingH.After - kpuPad };
            var padW = new Padding { Before = oldConv2D.PaddingW.Before - kpuPad, After = oldConv2D.PaddingW.After - kpuPad };

            var prePadH = GetPadding(padH, pre: true);
            var prePadW = GetPadding(padW, pre: true);
            var prePad = context.Graph.AddNode(new Pad(DataType.Float32, output.Shape, new[] { Padding.Zero, Padding.Zero, prePadH, prePadW }, 0.0f));
            var conv2d = context.Graph.AddNode(new KPUFakeConv2D(prePad.Output.Shape, oldConv2D.Groups == oldConv2D.Input.Shape[1], filterType, KPUPoolType.Pool_Bypass, oldConv2D.Weights, oldConv2D.Bias, oldConv2D.FusedActivation));
            var surPadH = GetPadding(padH, pre: false);
            var surPadW = GetPadding(padW, pre: false);
            var surPad = context.Graph.AddNode(new Pad(DataType.Float32, conv2d.Output.Shape, new[] { Padding.Zero, Padding.Zero, surPadH, surPadW }, 0.0f));
            var slice = context.Graph.AddNode(new StridedSlice(DataType.Float32, surPad.Output.Shape, new[] { 0, 0, 0, 0 }, new[] { 0, 0, 0, 0 }, new[] { 1, 1, oldConv2D.StrideH, oldConv2D.StrideW }, 15, 15, 0, 0, 0));
            prePad.Input.Connect(output);
            conv2d.Input.Connect(prePad.Output);
            surPad.Input.Connect(conv2d.Output);
            slice.Input.Connect(surPad.Output);

            foreach (var input in inputs.ToList())
                input.Connect(slice.Output);
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
