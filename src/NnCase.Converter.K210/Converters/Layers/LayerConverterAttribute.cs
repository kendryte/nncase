using System;
using System.Collections.Generic;
using System.Text;
using NnCase.Converter.K210.Converters.Stages.Inference;

namespace NnCase.Converter.K210.Converters.Layers
{
    [AttributeUsage(AttributeTargets.Class, AllowMultiple = true, Inherited = false)]
    public sealed class LayerConverterAttribute : Attribute
    {
        public Type Type { get; }

        public K210LayerType LayerType { get; }

        public LayerConverterAttribute(Type type, K210LayerType layerType)
        {
            Type = type;
            LayerType = layerType;
        }
    }
}
