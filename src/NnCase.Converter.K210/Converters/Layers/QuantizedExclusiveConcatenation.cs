using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NnCase.Converter.Converters;
using NnCase.Converter.K210.Converters.Stages.Convert;
using NnCase.Converter.K210.Converters.Stages.Inference;
using NnCase.Converter.K210.Converters.Stages.Quantize;
using NnCase.Converter.Model;
using NnCase.Converter.Model.Layers;

namespace NnCase.Converter.K210.Converters.Layers
{
    [LayerConverter(typeof(QuantizedExclusiveConcatenation), K210LayerType.Invalid)]
    public class QuantizedExclusiveConcatenationConverter
    {
        public void AllocateInputMemory(QuantizedExclusiveConcatenation layer, OutputConnector input, InferenceContext context)
        {
            var totalAlloc = context.GetOrAllocateMainMemory(layer.Output);
            uint offset = 0;

            foreach (var node in layer.Inputs.Select(x => x.Connection.From))
            {
                if (context.MainMemoryMap.ContainsKey(node))
                    return;
                uint size = (uint)node.Dimensions.GetSize();
                context.MainMemoryMap.Add(node, new MemoryAllocation(totalAlloc.Node, offset, size));
                offset += size;
            }
        }

        public void FixupQuantization(QuantizedExclusiveConcatenation layer, QuantizationContext context)
        {
            var outRange = context.Distributions[layer.Output];
            foreach (var input in layer.Inputs.Select(x => x.Connection.From))
                context.Distributions[input] = outRange;
        }
    }
}
