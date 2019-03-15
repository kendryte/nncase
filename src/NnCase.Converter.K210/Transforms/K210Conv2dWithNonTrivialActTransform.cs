using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NnCase.Converter.K210.Model.Layers;
using NnCase.Converter.Model;
using NnCase.Converter.Model.Layers;
using NnCase.Converter.Transforms;

namespace NnCase.Converter.K210.Transforms
{
    public class K210Conv2dWithNonTrivialActTransform : Transform
    {
        protected override bool OnTryMatch(Layer layer, TransformContext context)
        {
            try
            {
                if (layer is K210Conv2d conv2d && conv2d.FusedActivationFunction == ActivationFunctionType.Linear &&
                    conv2d.NonTrivialActivation == null)
                {
                    context.MatchedLayers.Add(layer);
                    context.Inputs.Add(conv2d.Input);

                    foreach (var nextLayer in conv2d.Output.Connections.Select(o => o.To.Owner))
                    {
                        if (nextLayer is Dequantize dequantize)
                        {
                            context.MatchedLayers.Add(nextLayer);

                            foreach (var nextLayer2 in dequantize.Output.Connections.Select(o => o.To.Owner))
                            {
                                if (nextLayer2 is LeakyRelu leaky)
                                {
                                    context.Outputs.Add(leaky.Output);
                                }
                                else
                                {
                                    continue;
                                }

                                context.MatchedLayers.Add(nextLayer2);
                                return true;
                            }
                        }
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
            var act = context.MatchedLayers[2];
            var input = conv2d.Input.Connection.From;
            var output = act.OutputConnectors[0];

            conv2d.Input.ClearConnection();

            var newConv2d = new K210Conv2d(conv2d.Input.Dimensions, conv2d.Conv2dType, conv2d.Weights, conv2d.Bias, conv2d.PoolType, conv2d.FusedActivationFunction, act);
            newConv2d.Input.SetConnection(input);
            var newDequantize = new Dequantize(newConv2d.Output.Dimensions);
            newDequantize.Input.SetConnection(newConv2d.Output);
            var oldOuts = output.Connections.Select(o => o.To).ToList();
            foreach (var oldOut in oldOuts)
                oldOut.SetConnection(newDequantize.Output);
        }
    }
}
