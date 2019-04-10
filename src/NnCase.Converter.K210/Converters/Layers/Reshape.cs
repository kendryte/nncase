using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NnCase.Converter.Converters;
using NnCase.Converter.K210.Converters.Stages.Convert;
using NnCase.Converter.K210.Converters.Stages.Inference;
using NnCase.Converter.Model;
using NnCase.Converter.Model.Layers;

namespace NnCase.Converter.K210.Converters.Layers
{
    [LayerConverter(typeof(Reshape), K210LayerType.Invalid)]
    public class ReshapeConverter
    {
        public void AllocateInputMemory(Reshape layer, OutputConnector input, InferenceContext context)
        {
            context.MainMemoryMap.Add(layer.Output, context.GetOrAllocateMainMemory(layer.Input.Connection.From));
        }
    }
}
