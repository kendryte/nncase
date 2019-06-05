using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using NnCase.Converter.K210.Converters.Stages.Convert;
using NnCase.Converter.K210.Converters.Stages.Generate;
using NnCase.Converter.K210.Converters.Stages.Inference;
using NnCase.Converter.K210.Emulator;
using NnCase.Converter.Model.Layers;

namespace NnCase.Converter.K210.Converters.Layers
{
    public class ChannelwiseDequantizeLayerArgument
    {
        public K210LayerFlags Flags { get; set; }

        public uint MainMemoryInputAddress { get; set; }

        public uint MainMemoryOutputAddress { get; set; }

        public uint Channels { get; set; }

        public uint ChannelSize { get; set; }

        public K210QuantizationParam[] QuantParams { get; set; }
    }

    [LayerConverter(typeof(ChannelwiseDequantize), K210LayerType.ChannelwiseDequantize)]
    public class ChannelwiseDequantizeConverter
    {
        public ChannelwiseDequantizeLayerArgument Convert(ChannelwiseDequantize layer, ConvertContext context)
        {
            var q = context.Quantization.Distributions[layer.Input.Connection.From];
            return new ChannelwiseDequantizeLayerArgument
            {
                Channels = (uint)layer.Input.Dimensions[1],
                ChannelSize = (uint)(layer.Input.Dimensions.GetSize() / layer.Input.Dimensions[1]),
                QuantParams = context.Quantization.Distributions[layer.Input.Connection.From].Channels.Select(x => x.GetQuantizationParam(8)).ToArray()
            };
        }

        public void Infer(ChannelwiseDequantize layer, ChannelwiseDequantizeLayerArgument argument, InferenceContext context)
        {
            var inputAlloc = context.MainMemoryMap[layer.Input.Connection.From];
            var outputAlloc = context.MainMemoryMap[layer.Output];

            argument.Flags = K210LayerFlags.MainMemoryOutput;
            argument.MainMemoryInputAddress = inputAlloc.GetAddress();
            argument.MainMemoryOutputAddress = outputAlloc.GetAddress();
        }

        public ChannelwiseDequantizeLayerArgument DeserializeBin(int offset, K210BinDeserializeContext context)
        {
            var sr = context.GetReaderAt(offset);
            var argument = new ChannelwiseDequantizeLayerArgument
            {
                Flags = sr.Read<K210LayerFlags>(),
                MainMemoryInputAddress = sr.Read<uint>(),
                MainMemoryOutputAddress = sr.Read<uint>(),
                Channels = sr.Read<uint>(),
                ChannelSize = sr.Read<uint>()
            };

            argument.QuantParams = sr.ReadArray<K210QuantizationParam>((int)argument.Channels);
            return argument;
        }

        public void Forward(ChannelwiseDequantizeLayerArgument argument, ForwardContext context)
        {
            var src = context.GetMainRamAt((int)argument.MainMemoryInputAddress);
            var dest = MemoryMarshal.Cast<byte, float>(context.GetMainRamAt((int)argument.MainMemoryOutputAddress));

            int idx = 0;
            for (int oc = 0; oc < argument.Channels; oc++)
            {
                var q = argument.QuantParams[oc];
                for (int i = 0; i < argument.ChannelSize; i++)
                {
                    dest[idx] = src[idx] * q.Scale + q.Bias;
                    idx++;
                }
            }
        }
    }
}
