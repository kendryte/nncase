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
    public class K210Conv2dToChannelwiseTransform : Transform
    {
        protected override bool OnTryMatch(Layer layer, TransformContext context)
        {
            try
            {
                if (layer is K210Conv2d conv2d && conv2d.NonTrivialActivation == null && !conv2d.IsChannelwiseOutput)
                {
                    context.MatchedLayers.Add(layer);
                    context.Inputs.Add(conv2d.Input);

                    foreach (var nextLayer in conv2d.Output.Connections.Select(o => o.To.Owner))
                    {
                        if (nextLayer is Dequantize dequantize && !dequantize.Output.Connections.Any(x => x.To.Owner is Quantize))
                        {
                            context.MatchedLayers.Add(nextLayer);
                            context.Outputs.Add(dequantize.Output);
                            return true;
                        }
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
            var conv2d = (K210Conv2d)context.MatchedLayers[0];
            var dequantize = (Dequantize)context.MatchedLayers[1];
            var output = dequantize.Output;

            conv2d.IsChannelwiseOutput = true;
            var channelwiseDeq = new ChannelwiseDequantize(conv2d.Output.Dimensions);
            channelwiseDeq.Input.SetConnection(conv2d.Output);
            var oldOuts = output.Connections.Select(o => o.To).ToList();
            foreach (var oldOut in oldOuts)
                oldOut.SetConnection(channelwiseDeq.Output);
        }
    }
}
