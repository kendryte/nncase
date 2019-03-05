using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics.Tensors;
using System.Text;
using NnCase.Converter.Model;
using NnCase.Converter.Model.Layers;

namespace NnCase.Converter.Transforms
{
    public class EliminateDequantizeOutputTransform : Transform
    {
        protected override bool OnTryMatch(Layer layer, TransformContext context)
        {
            try
            {
                if (layer is Dequantize dequantize)
                {
                    context.MatchedLayers.Add(layer);
                    context.Inputs.Add(dequantize.Input);

                    foreach (var nextLayer in dequantize.Output.Connections.Select(o => o.To.Owner))
                    {
                        if (nextLayer is OutputLayer outputLayer)
                        {
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
            var dequantize = (Dequantize)context.MatchedLayers[0];
            var outputLayer = (OutputLayer)context.MatchedLayers[1];
            var input = dequantize.Input.Connection.From;

            outputLayer.Input.SetConnection(input);
        }
    }
}
