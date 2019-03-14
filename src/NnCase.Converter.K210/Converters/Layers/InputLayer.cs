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
            if (context.DatasetProcess == Data.PostprocessMethods.Normalize0To1)
                context.Distributions[layer.Output] = new QuantizationRange { Min = 0, Max = 1 };
            else if (context.DatasetProcess == Data.PostprocessMethods.NormalizeMinus1To1)
                context.Distributions[layer.Output] = new QuantizationRange { Min = -1, Max = 1 };
        }
    }
}
