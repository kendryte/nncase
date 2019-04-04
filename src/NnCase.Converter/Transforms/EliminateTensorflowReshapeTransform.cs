using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NnCase.Converter.Model;
using NnCase.Converter.Model.Layers;

namespace NnCase.Converter.Transforms
{
    public class EliminateTensorflowReshapeTransform : Transform
    {
        protected override bool OnTryMatch(Layer layer, TransformContext context)
        {
            try
            {
                if (layer is TensorflowReshape reshape1)
                {
                    context.MatchedLayers.Add(layer);
                    context.Inputs.Add(reshape1.Input);

                    foreach (var nextLayer in reshape1.Output.Connections.Select(o => o.To.Owner))
                    {
                        if (nextLayer is TensorflowReshape reshape2 && reshape1.Input.Dimensions.SequenceEqual(reshape2.Output.Dimensions))
                        {
                            context.Outputs.Add(reshape2.Output);
                        }
                        else
                        {
                            continue;
                        }

                        context.MatchedLayers.Add(nextLayer);
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
            var reshape1 = (TensorflowReshape)context.MatchedLayers[0];
            var reshape2 = (TensorflowReshape)context.MatchedLayers[1];
            var input = reshape1.Input.Connection.From;
            var output = reshape2.Output;

            var oldOuts = output.Connections.Select(o => o.To).ToList();
            foreach (var oldOut in oldOuts)
                oldOut.SetConnection(input);
        }
    }
}