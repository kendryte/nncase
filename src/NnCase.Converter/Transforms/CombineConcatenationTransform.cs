using NnCase.Converter.Model;
using NnCase.Converter.Model.Layers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NnCase.Converter.Transforms
{
    public class CombineConcatenationTransform : Transform
    {
        protected override bool OnTryMatch(Layer layer, TransformContext context)
        {
            try
            {
                if (layer is Concatenation concat && concat.Output.Connections.Count == 1 && concat.Output.Connections[0].To.Owner is Concatenation concat2)
                {
                    context.Inputs.AddRange(concat.Inputs);
                    context.Inputs.AddRange(concat2.Inputs);
                    context.Outputs.Add(concat2.Output);

                    context.MatchedLayers.Add(concat);
                    context.MatchedLayers.Add(concat2);
                    return true;
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
            var concat1 = (Concatenation)context.MatchedLayers[0];
            var concat2 = (Concatenation)context.MatchedLayers[1];
            var output = concat2.Output;

            var inputs = new List<OutputConnector>();
            foreach (var item in concat2.Inputs.Select(x=>x.Connection.From))
            {
                if (item == concat1.Output)
                    inputs.AddRange(concat1.Inputs.Select(x => x.Connection.From));
                else
                    inputs.Add(item);
            }

            var newConcat = new Concatenation(inputs.Select(x => new ReadOnlyMemory<int>(x.Dimensions.ToArray())));
            for (int i = 0; i < inputs.Count; i++)
                newConcat.Inputs[i].SetConnection(inputs[i]);

            var oldOuts = output.Connections.Select(o => o.To).ToList();
            foreach (var oldOut in oldOuts)
                oldOut.SetConnection(newConcat.Output);
        }
    }
}
