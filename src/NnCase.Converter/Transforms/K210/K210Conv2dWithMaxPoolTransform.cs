using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NnCase.Converter.Model;
using NnCase.Converter.Model.Layers;
using NnCase.Converter.Model.Layers.K210;

namespace NnCase.Converter.Transforms.K210
{
    public class K210Conv2dWithMaxPoolTransform : Transform
    {
        protected override bool OnTryMatch(Layer layer, TransformContext context)
        {
            try
            {
                if (layer is K210Conv2d conv2d && conv2d.Conv2dType == K210Conv2dType.Conv2d)
                {
                    context.MatchedLayers.Add(layer);
                    context.Inputs.Add(conv2d.Input);

                    foreach (var nextLayer in conv2d.Output.Connections.Select(o => o.To.Owner))
                    {
                        if (nextLayer is MaxPool2d maxPool)
                        {
                            if (maxPool.FilterWidth != maxPool.FilterHeight ||
                                (maxPool.FilterWidth != 2 && maxPool.FilterHeight != 4) ||
                                maxPool.StrideWidth != maxPool.StrideHeight ||
                                maxPool.StrideWidth != maxPool.FilterWidth ||
                                maxPool.FusedActivationFunction != ActivationFunctionType.Linear)
                                continue;
                            context.Outputs.Add(maxPool.Output);
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
            var maxPool = (MaxPool2d)context.MatchedLayers[1];
            var input = conv2d.Input.Connection.From;
            var output = maxPool.Output;

            conv2d.Input.ClearConnection();

            var newConv2d = new K210Conv2d(conv2d.Input.Dimensions, conv2d.Conv2dType, conv2d.Weights, conv2d.Bias,
                maxPool.FilterWidth == 2 ? K210PoolType.MaxPool2x2 : K210PoolType.MaxPool4x4, conv2d.FusedActivationFunction);

            newConv2d.Input.SetConnection(input);
            var oldOuts = output.Connections.Select(o => o.To).ToList();
            foreach (var oldOut in oldOuts)
                oldOut.SetConnection(newConv2d.Output);
        }
    }
}
