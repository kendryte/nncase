using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NnCase.Converter.K210.Model.Layers;
using NnCase.Converter.Model;
using NnCase.Converter.Model.Layers;
using NnCase.Converter.Transforms;

namespace NnCase.Converter.K210.Transforms
{
    public class K210SeparableConv2dTransform : Transform
    {
        protected override bool OnTryMatch(Layer layer, TransformContext context)
        {
            try
            {
                if (layer is SpaceToBatchNd || layer is Pad)
                {
                    context.MatchedLayers.Add(layer);
                    context.Inputs.Add(layer.InputConnectors[0]);

                    foreach (var nextLayer in layer.OutputConnectors[0].Connections.Select(o => o.To.Owner))
                    {
                        if (nextLayer is DepthwiseConv2d dwConv2d)
                        {
                            if (dwConv2d.KernelWidth != 3 || dwConv2d.KernelHeight != 3 || dwConv2d.StrideHeight != 2 || dwConv2d.StrideWidth != 2 ||
                                dwConv2d.Padding != Padding.Valid)
                                return false;

                            context.MatchedLayers.Add(nextLayer);
                            context.Inputs.Add(dwConv2d.Input);

                            foreach (var nextLayer2 in dwConv2d.Output.Connections.Select(o => o.To.Owner))
                            {
                                if (nextLayer2 is Conv2d conv2d)
                                {
                                    if (conv2d.KernelWidth != 1 || conv2d.KernelHeight != 1 || conv2d.StrideHeight != 1 || conv2d.StrideWidth != 1)
                                        continue;
                                    context.Outputs.Add(conv2d.Output);
                                }
                                else
                                {
                                    continue;
                                }

                                context.MatchedLayers.Add(nextLayer2);
                                return true;
                            }
                        }
                        else
                        {
                            continue;
                        }

                        return false;
                    }
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
            var space = context.MatchedLayers[0];
            var dwConv2d = (DepthwiseConv2d)context.MatchedLayers[1];
            var conv2d = (Conv2d)context.MatchedLayers[2];
            var input = space.InputConnectors[0].Connection.From;
            var output = conv2d.Output;

            space.InputConnectors[0].ClearConnection();
            var newDwConv2d = new DepthwiseConv2d(input.Dimensions, dwConv2d.Weights, dwConv2d.Bias, Padding.Same, 1, 1, dwConv2d.FusedActivationFunction);
            var quantize = new Quantize(newDwConv2d.Output.Dimensions);
            var upload = new K210Upload(quantize.Output.Dimensions);
            var newConv2d = new K210Conv2d(upload.Output.Dimensions, K210Conv2dType.Conv2d, conv2d.Weights, conv2d.Bias, K210PoolType.LeftTop, conv2d.FusedActivationFunction, null);
            var dequantize = new Dequantize(newConv2d.Output.Dimensions);

            newDwConv2d.Input.SetConnection(input);
            quantize.Input.SetConnection(newDwConv2d.Output);
            upload.Input.SetConnection(quantize.Output);
            newConv2d.Input.SetConnection(upload.Output);
            dequantize.Input.SetConnection(newConv2d.Output);
            var oldOuts = output.Connections.Select(o => o.To).ToList();
            foreach (var oldOut in oldOuts)
                oldOut.SetConnection(dequantize.Output);
        }
    }
}
