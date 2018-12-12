using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NnCase.Converter.Model;
using NnCase.Converter.Model.Layers;
using NnCase.Converter.Model.Layers.K210;

namespace NnCase.Converter.Transforms.K210
{
    public class K210FullyConnectedTransform : Transform
    {
        protected override bool OnTryMatch(Layer layer, TransformContext context)
        {
            try
            {
                if (layer is FullyConnected fc)
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

            var addPad = new K210AddPadding(input.Dimensions);
            var conv2d = new K210Conv2d(addPad.Output.Dimensions, K210Conv2dType.Conv2d,
                fc.Weights.Reshape(new[] { fc.Weights.Dimensions[0], fc.Weights.Dimensions[1], 1, 1 }), fc.Bias, K210PoolType.None, fc.FusedActivationFunction);
            var removePad = new K210RemovePadding(conv2d.Output.Dimensions);

            addPad.Input.SetConnection(input);
            conv2d.Input.SetConnection(addPad.Output);
            removePad.Input.SetConnection(conv2d.Output);
            var oldOuts = output.Connections.Select(o => o.To).ToList();
            foreach (var oldOut in oldOuts)
                oldOut.SetConnection(removePad.Output);
        }
    }
}
