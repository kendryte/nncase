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
    public class K210SameConv2dTransform : Transform
    {
        protected override bool OnTryMatch(Layer layer, TransformContext context)
        {
            try
            {
                if (layer is Conv2d conv2d)
                {
                    if (conv2d.KernelWidth != conv2d.KernelHeight ||
                        (conv2d.KernelHeight != 1 && conv2d.KernelWidth != 3) ||
                        conv2d.StrideHeight != 1 || conv2d.StrideWidth != 1 ||
                        conv2d.Input.Dimensions[2] < 4 || conv2d.Input.Dimensions[3] < 4 ||
                        conv2d.Output.Dimensions[2] < 4 || conv2d.Output.Dimensions[3] < 4 ||
                        (conv2d.KernelWidth == 3 && conv2d.Padding != Padding.Same))
                        return false;
                    context.Inputs.Add(conv2d.Input);
                    context.Outputs.Add(conv2d.Output);
                }
                else if (layer is DepthwiseConv2d dwConv2d)
                {
                    if (dwConv2d.KernelWidth != dwConv2d.KernelHeight ||
                        (dwConv2d.KernelHeight != 1 && dwConv2d.KernelWidth != 3) ||
                        dwConv2d.StrideHeight != 1 || dwConv2d.StrideWidth != 1 ||
                        dwConv2d.Input.Dimensions[2] < 4 || dwConv2d.Input.Dimensions[3] < 4 ||
                        dwConv2d.Output.Dimensions[2] < 4 || dwConv2d.Output.Dimensions[3] < 4 ||
                        (dwConv2d.KernelWidth == 3 && dwConv2d.Padding != Padding.Same))
                        return false;
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
            K210Conv2d newLayer;
            OutputConnector output;
            OutputConnector input;
            var conv = context.MatchedLayers[0];
            if (conv is Conv2d conv2d)
            {
                newLayer = new K210Conv2d(conv2d.Input.Dimensions, K210Conv2dType.Conv2d, conv2d.Weights, conv2d.Bias, K210PoolType.None, conv2d.FusedActivationFunction, null);
                input = conv2d.Input.Connection.From;
                output = conv2d.Output;
                conv2d.Input.ClearConnection();
            }
            else if (conv is DepthwiseConv2d dwConv2d)
            {
                newLayer = new K210Conv2d(dwConv2d.Input.Dimensions, K210Conv2dType.DepthwiseConv2d, dwConv2d.Weights, dwConv2d.Bias, K210PoolType.None, dwConv2d.FusedActivationFunction, null);
                input = dwConv2d.Input.Connection.From;
                output = dwConv2d.Output;
                dwConv2d.Input.ClearConnection();
            }
            else
                throw new InvalidOperationException();

            var quantize = new Quantize(input.Dimensions);
            var upload = new K210Upload(input.Dimensions);
            var dequantize = new Dequantize(newLayer.Output.Dimensions);
            quantize.Input.SetConnection(input);
            upload.Input.SetConnection(quantize.Output);
            newLayer.Input.SetConnection(upload.Output);
            dequantize.Input.SetConnection(newLayer.Output);
            var oldOuts = output.Connections.Select(o => o.To).ToList();
            foreach (var oldOut in oldOuts)
                oldOut.SetConnection(dequantize.Output);
        }
    }
}
