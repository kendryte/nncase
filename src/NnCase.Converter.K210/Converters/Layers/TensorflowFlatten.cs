using System;
using System.Collections.Generic;
using System.Text;
using NnCase.Converter.K210.Converters.Stages.Convert;
using NnCase.Converter.K210.Converters.Stages.Inference;
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
    }
}
