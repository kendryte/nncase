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
    public class AveragePool2dLayerArgument
    {
        public K210LayerFlags Flags { get; set; }

        public uint MainMemoryInputAddress { get; set; }

        public uint MainMemoryOutputAddress { get; set; }

        public uint InputWidth { get; set; }

        public uint InputHeight { get; set; }

        public uint InputChannels { get; set; }

        public uint OutputWidth { get; set; }

        public uint OutputHeight { get; set; }

        public uint OutputChannels { get; set; }

        public uint KernelWidth { get; set; }

        public uint KernelHeight { get; set; }

        public uint StrideWidth { get; set; }

        public uint StrideHeight { get; set; }

        public uint PaddingWidth { get; set; }

        public uint PaddingHeight { get; set; }

        public ActivationFunctionType Activation { get; set; }
    }

    [LayerConverter(typeof(AveragePool2d), K210LayerType.AveragePool2d)]
    public class AveragePool2dConverter
    {
        public AveragePool2dLayerArgument Convert(AveragePool2d layer, ConvertContext context)
        {
            return new AveragePool2dLayerArgument
            {
                InputWidth = (uint)layer.Input.Dimensions[3],
                InputHeight = (uint)layer.Input.Dimensions[2],
                InputChannels = (uint)layer.Input.Dimensions[1],
                OutputWidth = (uint)layer.Output.Dimensions[3],
                OutputHeight = (uint)layer.Output.Dimensions[2],
                OutputChannels = (uint)layer.Output.Dimensions[1],
                KernelWidth = (uint)layer.FilterWidth,
                KernelHeight = (uint)layer.FilterHeight,
                StrideWidth = (uint)layer.StrideWidth,
                StrideHeight = (uint)layer.StrideHeight,
                PaddingWidth = (uint)Layer.GetPadding(layer.Input.Dimensions[3], layer.Output.Dimensions[3], layer.StrideWidth, 1, layer.FilterWidth),
                PaddingHeight = (uint)Layer.GetPadding(layer.Input.Dimensions[2], layer.Output.Dimensions[2], layer.StrideHeight, 1, layer.FilterHeight),
                Activation = layer.FusedActivationFunction
            };
        }

        public void Infer(AveragePool2d layer, AveragePool2dLayerArgument argument, InferenceContext context)
        {
            var inputAlloc = context.MainMemoryMap[layer.Input.Connection.From];
            var outputAlloc = context.MainMemoryMap[layer.Output];

            argument.Flags = K210LayerFlags.MainMemoryOutput;
            argument.MainMemoryInputAddress = inputAlloc.GetAddress();
            argument.MainMemoryOutputAddress = outputAlloc.GetAddress();
        }

        public AveragePool2dLayerArgument DeserializeBin(int offset, K210BinDeserializeContext context)
        {
            var sr = context.GetReaderAt(offset);
            var argument = new AveragePool2dLayerArgument
            {
                Flags = sr.Read<K210LayerFlags>(),
                MainMemoryInputAddress = sr.Read<uint>(),
                MainMemoryOutputAddress = sr.Read<uint>(),
                InputWidth = sr.Read<uint>(),
                InputHeight = sr.Read<uint>(),
                InputChannels = sr.Read<uint>(),
                OutputWidth = sr.Read<uint>(),
                OutputHeight = sr.Read<uint>(),
                OutputChannels = sr.Read<uint>(),
                KernelWidth = sr.Read<uint>(),
                KernelHeight = sr.Read<uint>(),
                StrideWidth = sr.Read<uint>(),
                StrideHeight = sr.Read<uint>(),
                PaddingWidth = sr.Read<uint>(),
                PaddingHeight = sr.Read<uint>(),
                Activation = sr.Read<ActivationFunctionType>()
            };

            return argument;
        }

        public void Forward(AveragePool2dLayerArgument argument, ForwardContext context)
        {
            var src = MemoryMarshal.Cast<byte, float>(context.GetMainRamAt((int)argument.MainMemoryInputAddress));
            var dest = MemoryMarshal.Cast<byte, float>(context.GetMainRamAt((int)argument.MainMemoryOutputAddress));

            int outIdx = 0;
            for (int oc = 0; oc < argument.OutputChannels; oc++)
            {
                var channelSrc = src.Slice((int)(argument.InputWidth * argument.InputHeight * oc));
                for (int oy = 0; oy < argument.OutputHeight; oy++)
                {
                    for (int ox = 0; ox < argument.OutputWidth; ox++)
                    {
                        int inXOrigin = (int)(ox * argument.StrideWidth) - (int)argument.PaddingWidth;
                        int inYOrigin = (int)(oy * argument.StrideHeight) - (int)argument.PaddingHeight;
                        int kernelXStart = Math.Max(0, -inXOrigin);
                        int kernelXEnd = Math.Min((int)argument.KernelWidth, (int)argument.InputWidth - inXOrigin);
                        int kernelYStart = Math.Max(0, -inYOrigin);
                        int kernelYEnd = Math.Min((int)argument.KernelHeight, (int)argument.InputHeight - inYOrigin);
                        float value = 0;
                        float kernelCount = 0;

                        for (int ky = kernelYStart; ky < kernelYEnd; ky++)
                        {
                            for (int kx = kernelXStart; kx < kernelXEnd; kx++)
                            {
                                int inX = inXOrigin + kx;
                                int inY = inYOrigin + ky;
                                value += channelSrc[inY * (int)argument.InputWidth + inX];
                                kernelCount++;
                            }
                        }

                        dest[outIdx++] = value / kernelCount;
                    }
                }
            }
        }
    }
}
