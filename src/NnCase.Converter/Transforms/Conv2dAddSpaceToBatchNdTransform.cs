using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics.Tensors;
using System.Text;
using NnCase.Converter.Model;
using NnCase.Converter.Model.Layers;

namespace NnCase.Converter.Transforms
{
    public class Conv2dAddSpaceToBatchNdTransform : Transform
    {
        protected override bool OnTryMatch(Layer layer, TransformContext context)
        {
            try
            {
                if (layer is Conv2d conv2d &&
                    conv2d.KernelWidth == 3 && conv2d.KernelHeight == 3 &&
                    conv2d.StrideWidth == 2 && conv2d.StrideHeight == 2 &&
                    conv2d.Padding == Padding.Same)
                {
                    context.Inputs.Add(conv2d.Input);
                    context.Outputs.Add(conv2d.Output);
                }
                else if (layer is DepthwiseConv2d dwConv2d &&
                    dwConv2d.KernelWidth == 3 && dwConv2d.KernelHeight == 3 &&
                    dwConv2d.StrideWidth == 2 && dwConv2d.StrideHeight == 2 &&
                    dwConv2d.Padding == Padding.Same)
                {
                    context.Inputs.Add(dwConv2d.Input);
                    context.Outputs.Add(dwConv2d.Output);
                }
                else
                {
                    return false;
                }

                context.MatchedLayers.Add(layer);
                return true;
            }
            catch
            {
                return false;
            }
        }

        public override void Process(TransformContext context)
        {
            if (context.MatchedLayers[0] is Conv2d conv2d)
            {
                var input = conv2d.Input.Connection.From;
                var output = conv2d.Output;

                conv2d.Input.ClearConnection();

                var space = new SpaceToBatchNd(input.Dimensions, new[] { 1, 1 }.ToTensor(), new[,] { { 1, 1 }, { 1, 1 } }.ToTensor());
                var newConv = new Conv2d(space.Output.Dimensions, conv2d.Weights, conv2d.Bias, Padding.Valid, conv2d.StrideWidth, conv2d.StrideHeight, conv2d.FusedActivationFunction);
                space.Input.SetConnection(input);
                newConv.Input.SetConnection(space.Output);

                var oldOuts = output.Connections.Select(o => o.To).ToList();
                foreach (var oldOut in oldOuts)
                    oldOut.SetConnection(newConv.Output);
            }
            else
            {
                var dwConv2d = (DepthwiseConv2d)context.MatchedLayers[0];
                var input = dwConv2d.Input.Connection.From;
                var output = dwConv2d.Output;

                dwConv2d.Input.ClearConnection();

                var space = new SpaceToBatchNd(input.Dimensions, new[] { 1, 1 }.ToTensor(), new[,] { { 1, 1 }, { 1, 1 } }.ToTensor());
                var newConv = new DepthwiseConv2d(space.Output.Dimensions, dwConv2d.Weights, dwConv2d.Bias, Padding.Valid, dwConv2d.StrideWidth, dwConv2d.StrideHeight, dwConv2d.FusedActivationFunction);
                space.Input.SetConnection(input);
                newConv.Input.SetConnection(space.Output);

                var oldOuts = output.Connections.Select(o => o.To).ToList();
                foreach (var oldOut in oldOuts)
                    oldOut.SetConnection(newConv.Output);
            }
        }
    }
}
