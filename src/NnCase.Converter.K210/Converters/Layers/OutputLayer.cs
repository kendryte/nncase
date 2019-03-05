using System;
using System.Collections.Generic;
using System.Text;
using NnCase.Converter.K210.Converters.Stages.Convert;
using NnCase.Converter.Model.Layers;

namespace NnCase.Converter.K210.Converters.Layers
{
    [LayerConverter(typeof(OutputLayer), Stages.Inference.K210LayerType.Invalid)]
    public class OutputLayerConverter
    {
    }
}
