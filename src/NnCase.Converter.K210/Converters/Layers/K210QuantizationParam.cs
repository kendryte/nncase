using System;
using System.Collections.Generic;
using System.Text;

namespace NnCase.Converter.K210.Converters.Layers
{
    public struct K210QuantizationParam
    {
        public float Scale { get; set; }

        public float Bias { get; set; }

        public override string ToString()
        {
            return $"{Scale}, {Bias}";
        }
    }
}
