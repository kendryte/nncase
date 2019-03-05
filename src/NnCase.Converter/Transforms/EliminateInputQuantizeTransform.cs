using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics.Tensors;
using System.Text;
using NnCase.Converter.Model;
using NnCase.Converter.Model.Layers;

namespace NnCase.Converter.Transforms
{
    public class EliminateInputQuantizeTransform : Transform
    {
        protected override bool OnTryMatch(Layer layer, TransformContext context)
        {
            try
            {
                if (layer is InputLayer inputLayer)
                {
                    context.MatchedLayers.Add(layer);

                    foreach (var nextLayer in inputLayer.Output.Connections.Select(o => o.To.Owner))
                    {
                        if (nextLayer is Quantize quantize)
                        {
                            context.Outputs.Add(quantize.Output);
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
            var inputLayer = (InputLayer)context.MatchedLayers[0];
            var quantize = (Quantize)context.MatchedLayers[1];
            var input = inputLayer.Output;
            var output = quantize.Output;

            var oldOuts = output.Connections.Select(o => o.To).ToList();
            foreach (var oldOut in oldOuts)
                oldOut.SetConnection(input);
        }
    }
}
