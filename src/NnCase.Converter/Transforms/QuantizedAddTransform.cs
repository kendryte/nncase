using NnCase.Converter.Model;
using NnCase.Converter.Model.Layers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NnCase.Converter.Transforms
{
    public class QuantizedAddTransform : Transform
    {
        protected override bool OnTryMatch(Layer layer, TransformContext context)
        {
            try
            {
                if (layer is Add add)
                {
                    if (add.InputA.Connection.From.Owner is Dequantize &&
                        add.InputB.Connection.From.Owner is Dequantize)
                    {
                        context.Inputs.Add(add.InputA);
                        context.Inputs.Add(add.InputB);
                        context.Outputs.Add(add.Output);

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
            var add = (Add)context.MatchedLayers[0];
            var inputA = add.InputA.Connection.From.Owner.InputConnectors[0].Connection.From;
            var inputB = add.InputB.Connection.From.Owner.InputConnectors[0].Connection.From;
            var output = add.Output;

            var quantAdd = new QuantizedAdd(add.InputA.Dimensions, add.InputB.Dimensions);
            var dequant = new Dequantize(quantAdd.Output.Dimensions);
            quantAdd.InputA.SetConnection(inputA);
            quantAdd.InputB.SetConnection(inputB);
            dequant.Input.SetConnection(quantAdd.Output);

            var oldOuts = output.Connections.Select(o => o.To).ToList();
            foreach (var oldOut in oldOuts)
                oldOut.SetConnection(dequant.Output);
        }
    }
}
