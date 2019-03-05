using System;
using System.Collections.Generic;
using System.Text;
using NnCase.Converter.K210.Converters.Stages.Quantize;
using NnCase.Converter.Model;

namespace NnCase.Converter.K210.Converters.Stages.Convert
{
    public class ConvertContext
    {
        public QuantizationContext Quantization { get; set; }

        public int WeightsBits { get; set; }

        public Dictionary<Layer, bool> ProcessMap { get; } = new Dictionary<Layer, bool>();

        public Dictionary<Layer, object> LayerArguments { get; } = new Dictionary<Layer, object>();
    }
}
