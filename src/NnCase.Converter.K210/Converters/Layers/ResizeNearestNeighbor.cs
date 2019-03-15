using System;
using System.Collections.Generic;
using System.Text;
using NnCase.Converter.K210.Converters.Stages.Convert;
using NnCase.Converter.K210.Converters.Stages.Inference;
using NnCase.Converter.Model.Layers;

namespace NnCase.Converter.K210.Converters.Layers
{
    public class ResizeNearestNeighborLayerArgument
    {
        public K210LayerFlags Flags { get; set; }

        public uint MainMemoryInputAddress { get; set; }

        public uint MainMemoryOutputAddress { get; set; }

        public uint InputWidth { get; set; }

        public uint InputHeight { get; set; }

        public uint Channels { get; set; }

        public uint OutputWidth { get; set; }

        public uint OutputHeight { get; set; }

        public int AlignCorners { get; set; }
    }

    [LayerConverter(typeof(ResizeNearestNeighbor), K210LayerType.ResizeNearestNeighbor)]
    public class ResizeNearestNeighborConverter
    {
        public ResizeNearestNeighborLayerArgument Convert(ResizeNearestNeighbor layer, ConvertContext context)
        {
            return new ResizeNearestNeighborLayerArgument
            {
                InputWidth = (uint)layer.Input.Dimensions[3],
                InputHeight = (uint)layer.Input.Dimensions[2],
                Channels = (uint)layer.Input.Dimensions[1],
                OutputWidth = (uint)layer.Output.Dimensions[3],
                OutputHeight = (uint)layer.Output.Dimensions[2],
                AlignCorners = layer.AlignCorners ? 1 : 0
            };
        }

        public void Infer(ResizeNearestNeighbor layer, ResizeNearestNeighborLayerArgument argument, InferenceContext context)
        {
            var inputAlloc = context.MainMemoryMap[layer.Input.Connection.From];
            var outputAlloc = context.MainMemoryMap[layer.Output];

            argument.MainMemoryInputAddress = inputAlloc.GetAddress();
            argument.MainMemoryOutputAddress = outputAlloc.GetAddress();
        }
    }
}
