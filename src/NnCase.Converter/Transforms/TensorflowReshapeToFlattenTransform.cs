using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NnCase.Converter.Model;
using NnCase.Converter.Model.Layers;

namespace NnCase.Converter.Transforms
{
    public class TensorflowReshapeToFlattenTransform : Transform
    {
        protected override bool OnTryMatch(Layer layer, TransformContext context)
        {
            try
            {
                if (layer is TensorflowReshape reshape &&
                    reshape.Input.Dimensions.GetSize() == reshape.Output.Dimensions[1] && reshape.Output.Dimensions.Length == 2)
                {
                    context.Inputs.Add(reshape.Input);
                    context.Outputs.Add(reshape.Output);
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
            var reshape = (TensorflowReshape)context.MatchedLayers[0];
            var input = reshape.Input.Connection.From;
            var output = reshape.Output;

            reshape.Input.ClearConnection();

            var flatten = new TensorflowFlatten(reshape.Input.Dimensions);
            flatten.Input.SetConnection(input);

            var oldOuts = output.Connections.Select(o => o.To).ToList();
            foreach (var oldOut in oldOuts)
                oldOut.SetConnection(flatten.Output);
        }
    }
}
