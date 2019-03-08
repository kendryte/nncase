using NnCase.Converter.Model;
using NnCase.Converter.Model.Layers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NnCase.Converter.Transforms
{
    public class QuantizedConcatenationTransform : Transform
    {
        protected override bool OnTryMatch(Layer layer, TransformContext context)
        {
            try
            {
                if (layer is Concatenation concat)
                {
                    if (concat.Inputs.Select(x => x.Connection.From.Owner).All(x => x is Dequantize))
                    {
                        context.Inputs.AddRange(concat.Inputs);
                        context.Outputs.Add(concat.Output);

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
            var concat = (Concatenation)context.MatchedLayers[0];
            var output = concat.Output;

            var exConcat = new QuantizedConcatenation(concat.Inputs.Select(x => new ReadOnlyMemory<int>(x.Dimensions.ToArray())));
            for (int i = 0; i < exConcat.Inputs.Count; i++)
            {
                var input = concat.Inputs[i].Connection.From;
                var quantize = new Quantize(input.Dimensions);
                var requantize = new Requantize(quantize.Output.Dimensions);
                quantize.Input.SetConnection(input);
                requantize.Input.SetConnection(quantize.Output);
                exConcat.Inputs[i].SetConnection(requantize.Output);
            }

            var dequantize = new Dequantize(exConcat.Output.Dimensions);
            dequantize.Input.SetConnection(exConcat.Output);

            var oldOuts = output.Connections.Select(o => o.To).ToList();
            foreach (var oldOut in oldOuts)
                oldOut.SetConnection(dequantize.Output);
        }
    }
}
