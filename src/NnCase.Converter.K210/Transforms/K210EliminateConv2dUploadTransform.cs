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
    public class K210EliminateConv2dUploadTransform : Transform
    {
        protected override bool SkipSelfContainedCheck => true;

        protected override bool OnTryMatch(Layer layer, TransformContext context)
        {
            try
            {
                if (layer is K210Conv2d conv2d)
                {
                    context.Inputs.Add(conv2d.Input);
                    context.MatchedLayers.Add(layer);

                    foreach (var nextLayer in conv2d.Output.Connections.Select(o => o.To.Owner))
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
            var conv2d = (K210Conv2d)context.MatchedLayers[0];
            var upload = (K210Upload)context.MatchedLayers[1];
            var input = conv2d.Output;
            var output = upload.Output;

            var oldOuts = output.Connections.Select(o => o.To).ToList();
            foreach (var oldOut in oldOuts)
                oldOut.SetConnection(input);
        }
    }
}
