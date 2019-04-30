using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NnCase.Converter.Model;
using NnCase.Converter.Model.Layers;

namespace NnCase.Converter.Transforms
{
    public class Conv2d1x1ToFullyConnectedTransform : Transform
    {
        protected override bool OnTryMatch(Layer layer, TransformContext context)
        {
            try
            {
                if (layer is Conv2d conv2d &&
                    conv2d.KernelWidth == 1 && conv2d.KernelHeight == 1 &&
                    conv2d.StrideWidth == 1 && conv2d.StrideHeight == 1 &&
                    conv2d.Input.Dimensions[2] == 1 && conv2d.Input.Dimensions[3] == 1)
                {
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
            var conv2d = (Conv2d)context.MatchedLayers[0];
            var input = conv2d.Input.Connection.From;
            var output = conv2d.Output;

            conv2d.Input.ClearConnection();

            var fc = new FullyConnected(input.Dimensions, conv2d.Weights.Reshape(new[] { conv2d.Weights.Dimensions[0], conv2d.Weights.Dimensions[1] }), conv2d.Bias, conv2d.FusedActivationFunction);
            var reshape = new Reshape(fc.Output.Dimensions, new[] { fc.Output.Dimensions[0], fc.Output.Dimensions[1], 1, 1 });
            fc.Input.SetConnection(input);
            reshape.Input.SetConnection(fc.Output);

            var oldOuts = output.Connections.Select(o => o.To).ToList();
            foreach (var oldOut in oldOuts)
                oldOut.SetConnection(reshape.Output);
        }
    }
}
