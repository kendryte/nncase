using NnCase.Converter.Model;
using NnCase.Converter.Model.Layers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NnCase.Converter.Transforms
{
    public class GlobalAveragePoolTransform : Transform
    {
        protected override bool OnTryMatch(Layer layer, TransformContext context)
        {
            try
            {
                if (layer is AveragePool2d avgPool)
                {
                    if (avgPool.FilterHeight != avgPool.Input.Dimensions[2] ||
                        avgPool.FilterWidth != avgPool.Input.Dimensions[3])
                        return false;
                    context.Inputs.Add(avgPool.Input);
                    context.Outputs.Add(avgPool.Output);
                }
                else
                {
                    return false;
                }

                context.MatchedLayers.Add(layer);
                return true;
            }
            catch
            {
                return false;
            }
        }

        public override void Process(TransformContext context)
        {
            var avgPool = (AveragePool2d)context.MatchedLayers[0];
            var input = avgPool.Input.Connection.From;
            var output = avgPool.Output;

            avgPool.Input.ClearConnection();

            var newAvg = new GlobalAveragePool(input.Dimensions);

            newAvg.Input.SetConnection(input);
            var oldOuts = output.Connections.Select(o => o.To).ToList();
            foreach (var oldOut in oldOuts)
                oldOut.SetConnection(newAvg.Output);
        }
    }
}
