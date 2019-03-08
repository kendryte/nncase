using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using NnCase.Converter.Converters;
using NnCase.Converter.K210.Converters.Layers;
using NnCase.Converter.K210.Converters.Stages.Quantize;
using NnCase.Converter.Model;
using NnCase.Converter.Model.Layers;

namespace NnCase.Converter.K210.Converters.Stages.Convert
{
    public static class Converter
    {
        public static ConvertContext Convert(Graph graph, QuantizationContext quantizationContext, int weightsBits)
        {
            var context = new ConvertContext { Quantization = quantizationContext, WeightsBits = weightsBits };
            var converters = (from t in typeof(Converter).Assembly.ExportedTypes
                              let attrs = t.GetCustomAttributes<LayerConverterAttribute>()
                              where attrs.Any()
                              from attr in attrs
                              select new
                              {
                                  Key = attr.Type,
                                  Value = new { Type = t, Method = t.GetMethod("Convert") }
                              }).ToDictionary(x => x.Key, x => x.Value);

            void ConvertLayer(Layer layer)
            {
                if (!context.ProcessMap.GetValueOrDefault(layer))
                {
                    context.ProcessMap[layer] = true;

                    var type = layer.GetType();
                    if (converters.TryGetValue(type, out var info))
                    {
                        if (info.Method != null)
                        {
                            var converter = Activator.CreateInstance(info.Type);
                            var layerArg = info.Method.Invoke(converter, new object[] { layer, context });
                            if (layerArg != null)
                                context.LayerArguments.Add(layer, layerArg);
                        }
                    }
                    else
                    {
                        throw new LayerNotSupportedException(type.Name);
                    }

                    foreach (var conn in layer.InputConnectors)
                    {
                        var nextLayer = conn.Connection?.From.Owner;
                        if (nextLayer != null)
                            ConvertLayer(nextLayer);
                    }
                }
            }

            foreach (var layer in graph.Outputs)
                ConvertLayer(layer);

            return context;
        }
    }
}
