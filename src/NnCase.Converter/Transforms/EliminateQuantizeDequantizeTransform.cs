using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics.Tensors;
using System.Text;
using NnCase.Converter.Model;
using NnCase.Converter.Model.Layers;
using NnCase.Converter.Model.Layers.K210;

namespace NnCase.Converter.Transforms.K210
{
    public class EliminateQuantizeDequantizeTransform : Transform
    {
        protected override bool OnTryMatch(Layer layer, TransformContext context)
        {
            try
            {
                if (layer is Quantize quantize)
                {
                    context.MatchedLayers.Add(layer);
                    context.Inputs.Add(quantize.Input);

                    foreach (var nextLayer in quantize.Output.Connections.Select(o => o.To.Owner))
                    {
                        if (nextLayer is Dequantize dequantize)
                        {
                            context.Outputs.Add(dequantize.Output);
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
                else if (layer is Dequantize dequantize)
                {
                    context.MatchedLayers.Add(layer);
                    context.Inputs.Add(dequantize.Input);

                    foreach (var nextLayer in dequantize.Output.Connections.Select(o => o.To.Owner))
                    {
                        if (nextLayer is Quantize quantize2)
                        {
                            context.Outputs.Add(quantize2.Output);
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
            var layer1 = context.MatchedLayers[0];
            var layer2 = context.MatchedLayers[1];
            var input = layer1.InputConnectors[0].Connection.From;
            var output = layer2.OutputConnectors[0];

            layer1.InputConnectors[0].ClearConnection();

            var oldOuts = output.Connections.Select(o => o.To).ToList();
            foreach (var oldOut in oldOuts)
                oldOut.SetConnection(input);
        }
    }
}
