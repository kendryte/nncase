using NnCase.Converter.Model;
using NnCase.Converter.Model.Layers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NnCase.Converter.Transforms
{
    public class ExclusiveConcatenationTransform : Transform
    {
        protected override bool OnTryMatch(Layer layer, TransformContext context)
        {
            try
            {
                if (layer is Concatenation concat)
                {
                    if (concat.Inputs.Select(x => x.Connection.From.Connections.Count).All(x => x == 1) &&
                        concat.Output.Connections.Select(x => x.To.Owner).All(x => !(x is ExclusiveConcatenation)))
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

            var exConcat = new ExclusiveConcatenation(concat.Inputs.Select(x => new ReadOnlyMemory<int>(x.Dimensions.ToArray())));
            for (int i = 0; i < exConcat.Inputs.Count; i++)
                exConcat.Inputs[i].SetConnection(concat.Inputs[i].Connection.From);

            var oldOuts = output.Connections.Select(o => o.To).ToList();
            foreach (var oldOut in oldOuts)
                oldOut.SetConnection(exConcat.Output);
        }
    }
}
