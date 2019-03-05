using System;
using System.Collections.Generic;
using System.Text;
using NnCase.Converter.K210.Converters.Stages.Convert;
using NnCase.Converter.Model.Layers;

namespace NnCase.Converter.K210.Converters.Layers
{
    public class QuantizedAddLayerArgument
    {
        public K210LayerFlags Flags { get; set; }

        public uint MainMemoryInputAAddress { get; set; }

        public uint MainMemoryInputBAddress { get; set; }

        public uint MainMemoryOutputAddress { get; set; }

        public uint Channels { get; set; }

        public uint InputAMul { get; set; }

        public uint InputAShift { get; set; }

        public uint InputBMul { get; set; }

        public uint InputBShift { get; set; }

        public int OutputOffset { get; set; }
    }

    [LayerConverter(typeof(QuantizedAdd), Stages.Inference.K210LayerType.Invalid)]
    public class QuantizedAddConverter
    {
        public QuantizedAddLayerArgument Convert(QuantizedAdd layer, ConvertContext context)
        {
            return new QuantizedAddLayerArgument
            {
                Channels = (uint)(layer.Output.Dimensions[1])
            };
        }
    }
}
