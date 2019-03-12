using System;
using System.Collections.Generic;
using System.Text;
using NnCase.Converter.K210.Converters.Stages.Convert;
using NnCase.Converter.K210.Converters.Stages.Inference;
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
    }
}
