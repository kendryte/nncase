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
    public class K210EliminateConv2dRequantizeTransform : Transform
    {
        protected override bool OnTryMatch(Layer layer, TransformContext context)
        {
            try
            {
                if (layer is K210Conv2d conv2d)
                {
                    context.Inputs.Add(conv2d.Input);
                    context.MatchedLayers.Add(layer);

                    foreach (var nextLayer in conv2d.Output.Connections.Select(o => o.To.Owner))
                    {
                        if (nextLayer is Requantize requantize)
                        {
                            context.Outputs.Add(requantize.Output);
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
            var conv2d = (K210Conv2d)context.MatchedLayers[0];
            var requantize = (Requantize)context.MatchedLayers[1];
            var input = conv2d.Output;
            var output = requantize.Output;

            var oldOuts = output.Connections.Select(o => o.To).ToList();
            foreach (var oldOut in oldOuts)
                oldOut.SetConnection(input);
        }
    }
}
