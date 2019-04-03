using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using NnCase.Converter.Converters;
using NnCase.Converter.K210.Converters.Layers;
using NnCase.Converter.K210.Converters.Stages.Inference;
using NnCase.Converter.Model;
using NnCase.Converter.Model.Layers;

namespace NnCase.Converter.K210.Converters.Stages.Generate
{
    public static class Generator
    {
        public static K210BinGenerationContext GenerateBin(Graph graph, Stream stream, int weightsBits, string prefix, InferenceContext inferenceContext)
        {
            var context = new K210BinGenerationContext
            {
                Prefix = prefix,
                MaxStartAddress = inferenceContext.KPUMemoryAllocator.MaxStart,
                MainMemoryUsage = inferenceContext.MainMemoryAllocator.MaxEnd,
                Outputs = (from o in graph.Outputs
                           let m = inferenceContext.MainMemoryMap[o.Input.Connection.From]
                           select new K210OutputAddress
                           {
                               Address = m.GetAddress(),
                               Size = (uint)o.Input.Dimensions.GetSize() * 4
                           }).ToList(),
                Stream = stream,
                WeightsBits = weightsBits
            };

            var converters = (from t in typeof(Generator).Assembly.ExportedTypes
                              let attrs = t.GetCustomAttributes<LayerConverterAttribute>()
                              where attrs.Any()
                              from attr in attrs
                              where attr.LayerType != K210LayerType.Invalid
                              select new
                              {
                                  Key = attr.LayerType,
                                  Value = new { Type = t, Method = t.GetMethod("GenerateBin") }
                              }).ToDictionary(x => x.Key, x => x.Value);

            var layers = inferenceContext.InferenceOrders;
            var bw = new BinaryWriter(stream);

            void GenerateBinLayerBody(K210Layer layer)
            {
                var type = layer.Header.Type;
                if (converters.TryGetValue(type, out var info))
                {
                    if (info.Method != null)
                    {
                        var converter = Activator.CreateInstance(info.Type);
                        info.Method.Invoke(converter, new object[] { bw, layer.Body, context });
                    }
                    else
                    {
                        GenerateBinDefault(bw, layer.Body);
                    }
                }
                else
                {
                    throw new LayerNotSupportedException(type.ToString());
                }

                context.AlignStreamPosition(8);
            }

            uint version = 3;
            uint flags = weightsBits == 8 ? 1u : 0u;
            bw.Write(version);
            bw.Write(flags);
            bw.Write(0);
            bw.Write(layers.Count);
            bw.Write(context.MaxStartAddress);
            bw.Write(context.MainMemoryUsage);

            // Outputs
            bw.Write(context.Outputs.Count);
            foreach (var output in context.Outputs)
            {
                bw.Write(output.Address);
                bw.Write(output.Size);
            }

            // Headers
            var fixPosition = bw.BaseStream.Position;
            bw.BaseStream.Position += 4 * 2 * layers.Count;

            for (int i = 0; i < layers.Count; i++)
            {
                var layer = layers[i];
                // BodySize
                var beginPosition = bw.BaseStream.Position;
                GenerateBinLayerBody(layer);
                layer.Header.BodySize = (uint)(bw.BaseStream.Position - beginPosition);
            }

            var newPosition = bw.BaseStream.Position;
            bw.BaseStream.Position = fixPosition;
            for (int i = 0; i < layers.Count; i++)
            {
                var header = layers[i].Header;
                bw.Write((uint)header.Type);
                bw.Write((uint)header.BodySize);
            }

            bw.BaseStream.Position = newPosition;
            return context;
        }

        private static void GenerateBinDefault(BinaryWriter bw, object argument)
        {
            var values = (from p in argument.GetType().GetProperties()
                          orderby p.MetadataToken
                          select p.GetValue(argument)).ToList();

            void WriteValue(object value)
            {
                switch (value)
                {
                    case uint v:
                        bw.Write(v);
                        break;
                    case int v:
                        bw.Write(v);
                        break;
                    case float v:
                        bw.Write(v);
                        break;
                    case K210LayerFlags v:
                        bw.Write((uint)v);
                        break;
                    case K210QuantizationParam v:
                        bw.Write(v.Scale);
                        bw.Write(v.Bias);
                        break;
                    case Padding v:
                        bw.Write((uint)v);
                        break;
                    case ActivationFunctionType v:
                        bw.Write((uint)v);
                        break;
                    case MemoryRange v:
                        bw.Write(v.Start);
                        bw.Write(v.Size);
                        break;
                    case byte[] v:
                        bw.Write(v);
                        break;
                    case IEnumerable v:
                        foreach (var i in v)
                            WriteValue(i);
                        break;
                    default:
                        throw new InvalidOperationException("Invalid argument member.");
                }
            }

            foreach (var value in values)
                WriteValue(value);
        }
    }
}
