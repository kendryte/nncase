using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics.Tensors;
using System.Text;
using NnCase.Converter.K210.Model.Layers;
using NnCase.Converter.Model;
using NnCase.Converter.Model.Layers;
using NnCase.Converter.Transforms;

namespace NnCase.Converter.K210.Transforms
{
    public class K210SpaceToBatchNdAndValidConv2dTransform : Transform
    {
        protected override bool OnTryMatch(Layer layer, TransformContext context)
        {
            try
            {
                if (layer is SpaceToBatchNd space)
                {
                    context.MatchedLayers.Add(layer);
                    context.Inputs.Add(space.Input);

                    foreach (var nextLayer in space.Output.Connections.Select(o => o.To.Owner))
                    {
                        if (nextLayer is Conv2d conv2d)
                        {
                            if (conv2d.KernelWidth != 3 || conv2d.KernelHeight != 3 ||
                                conv2d.Padding != Padding.Valid)
                                continue;
                            context.Outputs.Add(conv2d.Output);
                        }
                        else if (nextLayer is DepthwiseConv2d dwConv2d)
                        {
                            if (dwConv2d.KernelWidth != 3 || dwConv2d.KernelHeight != 3 || dwConv2d.StrideWidth != 1 || dwConv2d.StrideHeight != 1 ||
                                dwConv2d.Padding != Padding.Valid)
                                continue;
                            context.Outputs.Add(dwConv2d.Output);
                        }
                        else
                        {
                            continue;
                        }

                        context.MatchedLayers.Add(nextLayer);
                        return true;
                    }

                    return false;
                }

                return false;
            }
            catch
            {
                return false;
            }
        }

        public override void Process(TransformContext context)
        {
            var space = (SpaceToBatchNd)context.MatchedLayers[0];
            var input = space.Input.Connection.From;

            space.Input.ClearConnection();

            K210Conv2d newLayer;
            OutputConnector output;
            var conv = context.MatchedLayers[1];
            if (conv is Conv2d conv2d)
            {
                newLayer = new K210Conv2d(input.Dimensions, K210Conv2dType.Conv2d, conv2d.Weights, conv2d.Bias, conv2d.StrideWidth == 2 ? K210PoolType.LeftTop : K210PoolType.None, conv2d.FusedActivationFunction);
                output = conv2d.Output;
            }
            else if (conv is DepthwiseConv2d dwConv2d)
            {
                newLayer = new K210Conv2d(input.Dimensions, K210Conv2dType.DepthwiseConv2d, dwConv2d.Weights, dwConv2d.Bias, K210PoolType.None, dwConv2d.FusedActivationFunction);
                output = dwConv2d.Output;
            }
            else
                throw new InvalidOperationException();

            var quantize = new Quantize(input.Dimensions);
            var dequantize = new Dequantize(newLayer.Output.Dimensions);
            quantize.Input.SetConnection(input);
            newLayer.Input.SetConnection(quantize.Output);
            dequantize.Input.SetConnection(newLayer.Output);
            var oldOuts = output.Connections.Select(o => o.To).ToList();
            foreach (var oldOut in oldOuts)
                oldOut.SetConnection(dequantize.Output);
        }
    }
}
