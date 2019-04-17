using System;
using System.Collections.Generic;
using System.Text;
using NnCase.Converter.K210.Converters.Stages.Convert;
using NnCase.Converter.K210.Converters.Stages.Quantize;
using NnCase.Converter.Model.Layers;

namespace NnCase.Converter.K210.Converters.Layers
{
    [LayerConverter(typeof(InputLayer), Stages.Inference.K210LayerType.Invalid)]
    public class InputLayerConverter
    {
        public void FixupQuantization(InputLayer layer, QuantizationContext context)
        {
            var min = (0 - context.Mean) / context.Std;
            var max = (1 - context.Mean) / context.Std;
            context.Distributions[layer.Output] = new ChannelwiseRange(new QuantizationRange { Min = min, Max = max }, layer.Output.Dimensions[1]);
        }
    }
}
