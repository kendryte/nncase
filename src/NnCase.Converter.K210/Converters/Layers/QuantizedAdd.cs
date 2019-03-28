using System;
using System.Collections.Generic;
using System.Text;
using NnCase.Converter.K210.Converters.Stages.Convert;
using NnCase.Converter.K210.Converters.Stages.Inference;
using NnCase.Converter.K210.Converters.Stages.Quantize;
using NnCase.Converter.Model.Layers;

namespace NnCase.Converter.K210.Converters.Layers
{
    public class QuantizedAddLayerArgument
    {
        public K210LayerFlags Flags { get; set; }

        public uint MainMemoryInputAAddress { get; set; }

        public uint MainMemoryInputBAddress { get; set; }

        public uint MainMemoryOutputAddress { get; set; }

        public uint Count { get; set; }

        public int InputAOffset { get; set; }

        public int InputAMul { get; set; }

        public int InputAShift { get; set; }

        public int InputBOffset { get; set; }

        public int InputBMul { get; set; }

        public int InputBShift { get; set; }

        public int OutputOffset { get; set; }

        public int OutputMul { get; set; }

        public int OutputShift { get; set; }
    }

    [LayerConverter(typeof(QuantizedAdd), K210LayerType.QuantizedAdd)]
    public class QuantizedAddConverter
    {
        public QuantizedAddLayerArgument Convert(QuantizedAdd layer, ConvertContext context)
        {
            var inputARange = context.Quantization.Distributions[layer.InputA.Connection.From].Global;
            var inputBRange = context.Quantization.Distributions[layer.InputB.Connection.From].Global;
            var outputRange = context.Quantization.Distributions[layer.Output].Global;

            (var sa, var ba) = inputARange.GetScaleBias(8);
            (var sb, var bb) = inputBRange.GetScaleBias(8);
            (var so, var bo) = outputRange.GetScaleBias(8);

            (var mulA, var shiftA) = Quantizer.ExtractValueAndShift(sb, 32, 32);
            (var mulB, var shiftB) = Quantizer.ExtractValueAndShift(sa, 32, 32);
            (var mulO, var shiftO) = Quantizer.ExtractValueAndShift(so / (sa * sb), 32, 32);

            return new QuantizedAddLayerArgument
            {
                InputAOffset = (int)ba,
                InputAMul = (int)Math.Round(mulA),
                InputAShift = shiftA,
                InputBOffset = (int)bb,
                InputBMul = (int)Math.Round(mulB),
                InputBShift = shiftB,
                OutputOffset = (int)(-bo),
                OutputMul = (int)Math.Round(mulO),
                OutputShift = shiftO,
                Count = (uint)(layer.Output.Dimensions.GetSize())
            };
        }

        public void Infer(QuantizedAdd layer, QuantizedAddLayerArgument argument, InferenceContext context)
        {
            var inputAAlloc = context.MainMemoryMap[layer.InputA.Connection.From];
            var inputBAlloc = context.MainMemoryMap[layer.InputB.Connection.From];
            var outputAlloc = context.MainMemoryMap[layer.Output];

            argument.Flags = K210LayerFlags.MainMemoryOutput;
            argument.MainMemoryInputAAddress = inputAAlloc.GetAddress();
            argument.MainMemoryInputBAddress = inputBAlloc.GetAddress();
            argument.MainMemoryOutputAddress = outputAlloc.GetAddress();
        }
    }
}
