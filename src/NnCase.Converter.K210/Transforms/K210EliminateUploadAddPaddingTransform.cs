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
    public class K210EliminateUploadAddPaddingTransform : Transform
    {
        protected override bool OnTryMatch(Layer layer, TransformContext context)
        {
            try
            {
                if (layer is K210Upload upload)
                {
                    context.Inputs.Add(upload.Input);
                    context.MatchedLayers.Add(layer);

                    foreach (var nextLayer in upload.Output.Connections.Select(o => o.To.Owner))
                    {
                        if (nextLayer is K210AddPadding addPadding)
                        {
                            context.Outputs.Add(addPadding.Output);
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
            var upload = (K210Upload)context.MatchedLayers[0];
            var addPadding = (K210AddPadding)context.MatchedLayers[1];
            var input = upload.Input.Connection.From;

            var newAdd = new K210AddPadding(addPadding.Input.Dimensions);
            newAdd.Input.SetConnection(input);

            var oldOuts = addPadding.Output.Connections.Select(o => o.To).ToList();
            foreach (var oldOut in oldOuts)
                oldOut.SetConnection(newAdd.Output);
        }
    }
}
