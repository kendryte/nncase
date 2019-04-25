using NnCase.Converter.Model;
using NnCase.Converter.Model.Layers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NnCase.Converter.Transforms
{
    public class QuantizedResizeNearestNeighborTransform : Transform
    {
        protected override bool OnTryMatch(Layer layer, TransformContext context)
        {
            try
            {
                if (layer is ResizeNearestNeighbor resize)
                {
                    if (resize.Input.Connection.From.Owner is Dequantize)
                    {
                        context.Inputs.Add(resize.Input);
                        context.Outputs.Add(resize.Output);

                        context.MatchedLayers.Add(layer);
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
            var resize = (ResizeNearestNeighbor)context.MatchedLayers[0];
            var input = resize.Input.Connection.From.Owner.InputConnectors[0].Connection.From;
            var output = resize.Output;

            var quantResize = new QuantizedResizeNearestNeighbor(resize.Input.Dimensions, resize.Output.Dimensions[3], resize.Output.Dimensions[2], resize.AlignCorners);
            var dequant = new Dequantize(quantResize.Output.Dimensions);
            quantResize.Input.SetConnection(input);
            dequant.Input.SetConnection(quantResize.Output);

            var oldOuts = output.Connections.Select(o => o.To).ToList();
            foreach (var oldOut in oldOuts)
                oldOut.SetConnection(dequant.Output);
        }
    }
}
