using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics.Tensors;
using System.Text;
using NnCase.Converter.K210.Model.Layers;
using NnCase.Converter.Model;
using NnCase.Converter.Model.Layers;

namespace NnCase.Converter.Transforms
{
    public class K210EliminateQuantizedAddUploadTransform : Transform
    {
        protected override bool OnTryMatch(Layer layer, TransformContext context)
        {
            try
            {
                if (layer is QuantizedAdd add)
                {
                    context.Inputs.Add(add.InputA);
                    context.Inputs.Add(add.InputB);
                    context.MatchedLayers.Add(layer);

                    foreach (var nextLayer in add.Output.Connections.Select(o => o.To.Owner))
                    {
                        if (nextLayer is K210Upload upload)
                        {
                            context.Outputs.Add(upload.Output);
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
            var add = (QuantizedAdd)context.MatchedLayers[0];
            var upload = (K210Upload)context.MatchedLayers[1];
            var input = add.Output;
            var output = upload.Output;

            var oldOuts = output.Connections.Select(o => o.To).ToList();
            foreach (var oldOut in oldOuts)
                oldOut.SetConnection(input);
        }
    }
}
