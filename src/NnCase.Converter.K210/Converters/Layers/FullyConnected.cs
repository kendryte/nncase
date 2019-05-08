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
    public class FullyConnectedLayerArgument
    {
        public K210LayerFlags Flags { get; set; }

        public uint MainMemoryInputAddress { get; set; }

        public uint MainMemoryOutputAddress { get; set; }

        public uint InputChannels { get; set; }

        public uint OutputChannels { get; set; }

        public ActivationFunctionType Activation { get; set; }

        public float[] Weights { get; set; }

        public float[] Bias { get; set; }
    }

    [LayerConverter(typeof(FullyConnected), K210LayerType.FullyConnected)]
    public class FullyConnectedConverter
    {
        public FullyConnectedLayerArgument Convert(FullyConnected layer, ConvertContext context)
        {
            return new FullyConnectedLayerArgument
            {
                InputChannels = (uint)layer.Input.Dimensions[1],
                OutputChannels = (uint)layer.Output.Dimensions[1],
                Activation = layer.FusedActivationFunction,
                Weights = layer.Weights.ToArray(),
                Bias = layer.Bias.ToArray()
            };
        }

        public void Infer(FullyConnected layer, FullyConnectedLayerArgument argument, InferenceContext context)
        {
            var inputAlloc = context.MainMemoryMap[layer.Input.Connection.From];
            var outputAlloc = context.MainMemoryMap[layer.Output];

            argument.MainMemoryInputAddress = inputAlloc.GetAddress();
            argument.MainMemoryOutputAddress = outputAlloc.GetAddress();
        }

        public FullyConnectedLayerArgument DeserializeBin(int offset, K210BinDeserializeContext context)
        {
            var sr = context.GetReaderAt(offset);
            var argument = new FullyConnectedLayerArgument();
            argument.Flags = sr.Read<K210LayerFlags>();
            argument.MainMemoryInputAddress = sr.Read<uint>();
            argument.MainMemoryOutputAddress = sr.Read<uint>();
            argument.InputChannels = sr.Read<uint>();
            argument.OutputChannels = sr.Read<uint>();
            argument.Activation = sr.Read<ActivationFunctionType>();
            argument.Weights = MemoryMarshal.Cast<byte, float>(sr.ReadAsSpan((int)(argument.InputChannels * argument.OutputChannels * 4))).ToArray();
            argument.Bias = MemoryMarshal.Cast<byte, float>(sr.ReadAsSpan((int)(argument.OutputChannels * 4))).ToArray();

            return argument;
        }

        public void Forward(FullyConnectedLayerArgument argument, ForwardContext context)
        {
            var src = MemoryMarshal.Cast<byte, float>(context.GetMainRamAt((int)argument.MainMemoryInputAddress));
            var dest = MemoryMarshal.Cast<byte, float>(context.GetMainRamAt((int)argument.MainMemoryOutputAddress));

            for (int oc = 0; oc < argument.OutputChannels; oc++)
            {
                var weights = new ReadOnlySpan<float>(argument.Weights, (int)(oc * argument.InputChannels), (int)argument.InputChannels);
                float sum = 0;
                for (int ic = 0; ic < argument.InputChannels; ic++)
                    sum += src[ic] * weights[ic];
                dest[oc] = sum + argument.Bias[oc];
            }
        }
    }
}
