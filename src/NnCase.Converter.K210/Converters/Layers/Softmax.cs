using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using NnCase.Converter.K210.Converters.Stages.Convert;
using NnCase.Converter.K210.Converters.Stages.Generate;
using NnCase.Converter.K210.Converters.Stages.Inference;
using NnCase.Converter.K210.Emulator;
using NnCase.Converter.Model.Layers;

namespace NnCase.Converter.K210.Converters.Layers
{
    public class SoftmaxLayerArgument
    {
        public K210LayerFlags Flags { get; set; }

        public uint MainMemoryInputAddress { get; set; }

        public uint MainMemoryOutputAddress { get; set; }

        public uint Channels { get; set; }
    }

    [LayerConverter(typeof(Softmax), K210LayerType.Softmax)]
    public class SoftmaxConverter
    {
        public SoftmaxLayerArgument Convert(Softmax layer, ConvertContext context)
        {
            return new SoftmaxLayerArgument
            {
                Channels = (uint)layer.Input.Dimensions[1]
            };
        }

        public void Infer(Softmax layer, SoftmaxLayerArgument argument, InferenceContext context)
        {
            var inputAlloc = context.MainMemoryMap[layer.Input.Connection.From];
            var outputAlloc = context.MainMemoryMap[layer.Output];

            argument.Flags = K210LayerFlags.MainMemoryOutput;
            argument.MainMemoryInputAddress = inputAlloc.GetAddress();
            argument.MainMemoryOutputAddress = outputAlloc.GetAddress();
        }

        public SoftmaxLayerArgument DeserializeBin(int offset, K210BinDeserializeContext context)
        {
            var sr = context.GetReaderAt(offset);
            var argument = new SoftmaxLayerArgument
            {
                Flags = sr.Read<K210LayerFlags>(),
                MainMemoryInputAddress = sr.Read<uint>(),
                MainMemoryOutputAddress = sr.Read<uint>(),
                Channels = sr.Read<uint>()
            };

            return argument;
        }

        public void Forward(SoftmaxLayerArgument argument, ForwardContext context)
        {
            var src = MemoryMarshal.Cast<byte, float>(context.GetMainRamAt((int)argument.MainMemoryInputAddress));
            var dest = MemoryMarshal.Cast<byte, float>(context.GetMainRamAt((int)argument.MainMemoryOutputAddress));

            float max = float.MinValue;
            for (int oc = 0; oc < argument.Channels; oc++)
                max = Math.Max(max, src[oc]);

            float sum = 0;
            for (int oc = 0; oc < argument.Channels; oc++)
            {
                float value = (float)Math.Exp(src[oc] - max);
                sum += value;
                dest[oc] = value;
            }

            for (int oc = 0; oc < argument.Channels; oc++)
                dest[oc] /= sum;
        }
    }
}
