using NnCase.Converter.Model;
using NnCase.Converter.Model.Layers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NnCase.Converter.Transforms
{
    public class QuantizedMaxPool2dTransform : Transform
    {
        protected override bool OnTryMatch(Layer layer, TransformContext context)
        {
            try
            {
                if (layer is MaxPool2d maxPool)
                {
                    if (maxPool.Input.Connection.From.Owner is Dequantize)
                    {
                        context.Inputs.Add(maxPool.Input);
                        context.Outputs.Add(maxPool.Output);

                        context.MatchedLayers.Add(layer);
                        return true;
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
            var maxPool = (MaxPool2d)context.MatchedLayers[0];
            var input = maxPool.Input.Connection.From.Owner.InputConnectors[0].Connection.From;
            var output = maxPool.Output;

            var quantMaxPool = new QuantizedMaxPool2d(maxPool.Input.Dimensions, maxPool.Padding, maxPool.FilterWidth, maxPool.FilterHeight, maxPool.StrideWidth, maxPool.StrideHeight, maxPool.FusedActivationFunction);
            var dequant = new Dequantize(quantMaxPool.Output.Dimensions);
            quantMaxPool.Input.SetConnection(input);
            dequant.Input.SetConnection(quantMaxPool.Output);

            var oldOuts = output.Connections.Select(o => o.To).ToList();
            foreach (var oldOut in oldOuts)
                oldOut.SetConnection(dequant.Output);
        }
    }
}
