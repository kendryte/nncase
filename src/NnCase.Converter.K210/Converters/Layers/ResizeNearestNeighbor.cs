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

        public ResizeNearestNeighborLayerArgument DeserializeBin(int offset, K210BinDeserializeContext context)
        {
            var sr = context.GetReaderAt(offset);
            var argument = new ResizeNearestNeighborLayerArgument
            {
                Flags = sr.Read<K210LayerFlags>(),
                MainMemoryInputAddress = sr.Read<uint>(),
                MainMemoryOutputAddress = sr.Read<uint>(),
                InputWidth = sr.Read<uint>(),
                InputHeight = sr.Read<uint>(),
                Channels = sr.Read<uint>(),
                OutputWidth = sr.Read<uint>(),
                OutputHeight = sr.Read<uint>(),
                AlignCorners = sr.Read<int>()
            };

            return argument;
        }

        public void Forward(ResizeNearestNeighborLayerArgument argument, ForwardContext context)
        {
            var src = MemoryMarshal.Cast<byte, float>(context.GetMainRamAt((int)argument.MainMemoryInputAddress));
            var dest = MemoryMarshal.Cast<byte, float>(context.GetMainRamAt((int)argument.MainMemoryOutputAddress));

            float heightScale = (float)argument.InputHeight / argument.OutputHeight;
            float widthScale = (float)argument.InputWidth / argument.OutputWidth;

            int destIdx = 0;
            for (int oc = 0; oc < argument.Channels; oc++)
            {
                var channelSrc = src.Slice((int)(argument.InputWidth * argument.InputHeight * oc));
                for (int oy = 0; oy < argument.OutputHeight; oy++)
                {
                    var inY = (int)Math.Min(Math.Floor(oy * heightScale), argument.InputHeight - 1);
                    var yOrigin = channelSrc.Slice(inY * (int)argument.InputWidth);
                    for (int ox = 0; ox < argument.OutputWidth; ox++)
                    {
                        var inX = (int)Math.Min(Math.Floor(ox * widthScale), argument.InputWidth - 1);
                        dest[destIdx++] = yOrigin[inX];
                    }
                }
            }
        }
    }
}
