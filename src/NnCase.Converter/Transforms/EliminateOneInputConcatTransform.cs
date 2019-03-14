using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NnCase.Converter.Model;
using NnCase.Converter.Model.Layers;

namespace NnCase.Converter.Transforms
{
    public class EliminateOneInputConcatTransform : Transform
    {
        protected override bool OnTryMatch(Layer layer, TransformContext context)
        {
            try
            {
                if (layer is Concatenation concat && concat.Inputs.Count == 1)
                {
                    context.MatchedLayers.Add(layer);
                    context.Inputs.Add(concat.Inputs[0]);
                    context.Outputs.Add(concat.Output);
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
            var input = concat.Inputs[0].Connection.From;
            var output = concat.Output;

            var oldOuts = output.Connections.Select(o => o.To).ToList();
            foreach (var oldOut in oldOuts)
                oldOut.SetConnection(input);
        }
    }
}