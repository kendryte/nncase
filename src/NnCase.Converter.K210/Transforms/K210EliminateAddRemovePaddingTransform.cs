using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics.Tensors;
using System.Text;
using NnCase.Converter.K210.Model.Layers;
using NnCase.Converter.Model;
using NnCase.Converter.Model.Layers;
using NnCase.Converter.Transforms;

namespace NnCase.Converter.K210.Transforms
{
    public class K210EliminateAddRemovePaddingTransform : Transform
    {
        protected override bool OnTryMatch(Layer layer, TransformContext context)
        {
            try
            {
                if (layer is K210AddPadding addPadding)
                {
                    context.MatchedLayers.Add(layer);
                    context.Inputs.Add(addPadding.Input);

                    foreach (var nextLayer in addPadding.Output.Connections.Select(o => o.To.Owner))
                    {
                        if (nextLayer is K210RemovePadding removePadding)
                        {
                            context.Outputs.Add(removePadding.Output);
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
                else if (layer is K210RemovePadding removePadding)
                {
                    context.MatchedLayers.Add(layer);
                    context.Inputs.Add(removePadding.Input);

                    foreach (var nextLayer in removePadding.Output.Connections.Select(o => o.To.Owner))
                    {
                        if (nextLayer is K210AddPadding addPadding2)
                        {
                            context.Outputs.Add(addPadding2.Output);
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
