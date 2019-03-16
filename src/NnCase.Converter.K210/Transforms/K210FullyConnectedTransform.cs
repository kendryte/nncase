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
    public class K210FullyConnectedTransform : Transform
    {
        protected override bool OnTryMatch(Layer layer, TransformContext context)
        {
            try
            {
                if (layer is FullyConnected fc && fc.Input.Dimensions[1] <= 1024 && fc.Output.Dimensions[1] <= 1024)
                {
                    context.Inputs.Add(fc.Input);
                    context.Outputs.Add(fc.Output);
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
            var fc = (FullyConnected)context.MatchedLayers[0];
            var input = fc.Input.Connection.From;
            var output = fc.Output;

            fc.Input.ClearConnection();

            var quantize = new Quantize(input.Dimensions);
            var upload = new K210Upload(input.Dimensions);
            var addPad = new K210AddPadding(input.Dimensions);
            var conv2d = new K210Conv2d(addPad.Output.Dimensions, K210Conv2dType.Conv2d,
                fc.Weights.Reshape(new[] { fc.Weights.Dimensions[0], fc.Weights.Dimensions[1], 1, 1 }), fc.Bias, K210PoolType.None, fc.FusedActivationFunction, null);
            var removePad = new K210RemovePadding(conv2d.Output.Dimensions);
            var dequantize = new Dequantize(removePad.Output.Dimensions);

            quantize.Input.SetConnection(input);
            upload.Input.SetConnection(quantize.Output);
            addPad.Input.SetConnection(upload.Output);
            conv2d.Input.SetConnection(addPad.Output);
            removePad.Input.SetConnection(conv2d.Output);
            dequantize.Input.SetConnection(removePad.Output);
            var oldOuts = output.Connections.Select(o => o.To).ToList();
            foreach (var oldOut in oldOuts)
                oldOut.SetConnection(dequantize.Output);
        }
    }
}
