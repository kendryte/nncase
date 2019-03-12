using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NnCase.Converter.Model;
using NnCase.Converter.Model.Layers;

namespace NnCase.Converter.Transforms
{
    public class LeakyReluTransform : Transform
    {
        protected override bool OnTryMatch(Layer layer, TransformContext context)
        {
            try
            {
                if (layer is Mul mul && mul.Scale is float)
                {
                    context.Inputs.Add(mul.Input);
                    context.MatchedLayers.Add(layer);

                    foreach (var nextLayer in mul.Output.Connections.Select(o => o.To.Owner))
                    {
                        if (nextLayer is Maximum maximum)
                        {
                            if (maximum.InputA.Connection.From == mul.Input.Connection.From)
                            {
                                context.Inputs.Add(maximum.InputA);
                                context.Outputs.Add(maximum.Output);
                                context.MatchedLayers.Add(nextLayer);
                                return true;
                            }
                            else if (maximum.InputB.Connection.From == mul.Input.Connection.From)
                            {
                                context.Inputs.Add(maximum.InputB);
                                context.Outputs.Add(maximum.Output);
                                context.MatchedLayers.Add(nextLayer);
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
            var mul = (Mul)context.MatchedLayers[0];
            var max = (Maximum)context.MatchedLayers[1];
            var input = mul.Input.Connection.From;
            var output = max.Output;

            var leaky = new LeakyRelu(input.Dimensions, (float)mul.Scale);
            leaky.Input.SetConnection(input);

            var oldOuts = output.Connections.Select(o => o.To).ToList();
            foreach (var oldOut in oldOuts)
                oldOut.SetConnection(leaky.Output);
        }
    }
}
