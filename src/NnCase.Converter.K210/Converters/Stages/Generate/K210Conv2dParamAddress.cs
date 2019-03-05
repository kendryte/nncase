using System;
using System.Collections.Generic;
using System.Text;

namespace NnCase.Converter.K210.Converters.Stages.Generate
{
    public class K210Conv2dParamAddress
    {
        public uint Layer { get; set; }

        public uint Weights { get; set; }

        public uint Bn { get; set; }

        public uint Activation { get; set; }
    }
}
