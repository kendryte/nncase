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
    public class K210Stride2Conv2dTransform : Transform
    {
        protected override bool OnTryMatch(Layer layer, TransformContext context)
        {
            try
            {
                if (layer is Conv2d conv2d)
                {
                    if (conv2d.KernelWidth != conv2d.KernelHeight ||
                        (conv2d.KernelWidth != 3) || conv2d.Input.Dimensions[2] % 2 != 0 || conv2d.Input.Dimensions[3] % 2 != 0 ||
                        conv2d.StrideHeight != 2 || conv2d.StrideWidth != 2 ||
                        conv2d.Padding != Padding.Same)
                        return false;
                    context.Inputs.Add(conv2d.Input);
                    context.Outputs.Add(conv2d.Output);
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
                newLayer = new K210Conv2d(conv2d.Input.Dimensions, K210Conv2dType.Conv2d, conv2d.Weights, conv2d.Bias, K210PoolType.LeftTop, conv2d.FusedActivationFunction, null);
                input = conv2d.Input.Connection.From;
                output = conv2d.Output;
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
