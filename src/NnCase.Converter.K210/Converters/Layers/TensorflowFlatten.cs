using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using NnCase.Converter.K210.Converters.Stages.Convert;
using NnCase.Converter.K210.Converters.Stages.Generate;
using NnCase.Converter.K210.Converters.Stages.Inference;
using NnCase.Converter.K210.Emulator;
using NnCase.Converter.Model;
using NnCase.Converter.Model.Layers;

namespace NnCase.Converter.K210.Converters.Layers
{
    public class TensorflowFlattenLayerArgument
    {
        public K210LayerFlags Flags { get; set; }

        public uint MainMemoryInputAddress { get; set; }

        public uint MainMemoryOutputAddress { get; set; }

        public uint Width { get; set; }

        public uint Height { get; set; }

        public uint Channels { get; set; }
    }

    [LayerConverter(typeof(TensorflowFlatten), K210LayerType.TensorflowFlatten)]
    public class TensorflowFlattenConverter
    {
        public TensorflowFlattenLayerArgument Convert(TensorflowFlatten layer, ConvertContext context)
        {
            return new TensorflowFlattenLayerArgument
            {
                Width = (uint)layer.Input.Dimensions[3],
                Height = (uint)layer.Input.Dimensions[2],
                Channels = (uint)layer.Input.Dimensions[1]
            };
        }

        public void Infer(TensorflowFlatten layer, TensorflowFlattenLayerArgument argument, InferenceContext context)
        {
            var inputAlloc = context.MainMemoryMap[layer.Input.Connection.From];
            var outputAlloc = context.MainMemoryMap[layer.Output];

            argument.Flags = K210LayerFlags.MainMemoryOutput;
            argument.MainMemoryInputAddress = inputAlloc.GetAddress();
            argument.MainMemoryOutputAddress = outputAlloc.GetAddress();
        }

        public TensorflowFlattenLayerArgument DeserializeBin(int offset, K210BinDeserializeContext context)
        {
            var sr = context.GetReaderAt(offset);
            var argument = new TensorflowFlattenLayerArgument();
            argument.Flags = sr.Read<K210LayerFlags>();
            argument.MainMemoryInputAddress = sr.Read<uint>();
            argument.MainMemoryOutputAddress = sr.Read<uint>();
            argument.Width = sr.Read<uint>();
            argument.Height = sr.Read<uint>();
            argument.Channels = sr.Read<uint>();

            return argument;
        }

        public void Forward(TensorflowFlattenLayerArgument argument, ForwardContext context)
        {
            var src = MemoryMarshal.Cast<byte, float>(context.GetMainRamAt((int)argument.MainMemoryInputAddress));
            var dest = MemoryMarshal.Cast<byte, float>(context.GetMainRamAt((int)argument.MainMemoryOutputAddress));

            int i = 0;
            for (int oy = 0; oy < argument.Height; oy++)
            {
                for (int ox = 0; ox < argument.Width; ox++)
                {
                    for (int oc = 0; oc < argument.Channels; oc++)
                    {
                        dest[i++] = src[(int)((oc * argument.Height + oy) * argument.Width + ox)];
                    }
                }
            }
        }
    }
}
