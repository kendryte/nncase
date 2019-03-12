using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NnCase.Converter.K210.Converters.Stages.Convert;
using NnCase.Converter.K210.Converters.Stages.Inference;
using NnCase.Converter.Model.Layers;

namespace NnCase.Converter.K210.Converters.Layers
{
    [LayerConverter(typeof(QuantizedConcatenation), K210LayerType.QuantizedConcatenation)]
    public class QuantizedConcatenationConverter
    {
        public ConcatenationLayerArgument Convert(QuantizedConcatenation layer, ConvertContext context)
        {
            return new ConcatenationLayerArgument
            {
                InputCount = (uint)layer.Inputs.Count
            };
        }

        public void Infer(QuantizedConcatenation layer, ConcatenationLayerArgument argument, InferenceContext context)
        {
            var outputAlloc = context.MainMemoryMap[layer.Output];
        
            argument.Flags = K210LayerFlags.MainMemoryOutput;
            argument.MainMemoryOutputAddress = outputAlloc.GetAddress();
            argument.InputsMainMemory = (from i in layer.Inputs
                                         let a = context.MainMemoryMap[i.Connection.From]
                                         select new MemoryRange
                                         {
                                             Start = a.GetAddress(),
                                             Size = a.Size
                                         }).ToList();
        }
    }
}
