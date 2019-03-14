using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using NnCase.Converter.Converters;
using NnCase.Converter.K210.Converters.Layers;
using NnCase.Converter.K210.Model.Layers;
using NnCase.Converter.Model;
using NnCase.Converter.Model.Layers;

namespace NnCase.Converter.K210.Converters.Stages.Inference
{
    public static class InferExecutor
    {
        public static InferenceContext Infer(Graph graph, Convert.ConvertContext convertContext)
        {
            var context = new InferenceContext { };
            var converters = (from t in typeof(InferExecutor).Assembly.ExportedTypes
                              let attrs = t.GetCustomAttributes<LayerConverterAttribute>()
                              where attrs.Any()
                              from attr in attrs
                              select new
                              {
                                  Key = attr.Type,
                                  Value = new
                                  {
                                      Type = t,
                                      Layer = attr.LayerType,
                                      AllocMethod = t.GetMethod("AllocateInputMemory"),
                                      InferMethod = t.GetMethod("Infer")
                                  }
                              }).ToDictionary(x => x.Key, x => x.Value);

            void InferLayer(Layer layer)
            {
                if (!context.ProcessMap.GetValueOrDefault(layer))
                {
                    context.ProcessMap[layer] = true;

                    foreach (var conn in layer.InputConnectors)
                    {
                        var inputLayer = conn.Connection?.From.Owner;
                        if (inputLayer != null)
                            InferLayer(inputLayer);
                    }

                    foreach (var output in layer.OutputConnectors)
                    {
                        foreach (var conn in output.Connections.Select(x => x.To.Owner))
                        {
                            if (converters.TryGetValue(conn.GetType(), out var info) && info.AllocMethod != null)
                            {
                                var converter = Activator.CreateInstance(info.Type);
                                info.AllocMethod.Invoke(converter, new object[] { conn, output, context });
                            }
                            else
                            {
                                AllocateInputMemoryDefault(conn, output, context);
                            }
                        }
                    }

                    {
                        var type = layer.GetType();
                        if (converters.TryGetValue(type, out var info))
                        {
                            if (info.InferMethod != null)
                            {
                                var converter = Activator.CreateInstance(info.Type);
                                var argument = convertContext.LayerArguments[layer];
                                info.InferMethod.Invoke(converter, new object[] { layer, argument, context });
                                context.InferenceOrders.Add(new K210Layer { Header = new K210LayerHeader { Type = info.Layer }, Body = argument });
                            }
                        }
                        else
                        {
                            throw new LayerNotSupportedException(type.Name);
                        }
                    }

                    Console.Write($"{context.InferenceId++}: {layer.GetType().Name}");
                    if (layer.InputConnectors.Count != 0)
                        Console.Write($" {string.Join("x", layer.InputConnectors[0].Dimensions.ToArray())}");
                    if (layer.OutputConnectors.Count != 0)
                        Console.Write($" -> {string.Join("x", layer.OutputConnectors[0].Dimensions.ToArray())}");
                    Console.WriteLine();

                    if (!(layer is OutputLayer))
                    {
                        foreach (var conn in layer.InputConnectors)
                        {
                            var output = conn.Connection?.From;
                            if (output != null)
                            {
                                if (context.KPUMemoryMap.TryGetValue(output, out var alloc) && alloc.Node.IsUsed)
                                    alloc.Node.Release();
                                if (!(layer is K210Conv2d) && context.MainMemoryMap.TryGetValue(output, out var alloc2))
                                    alloc2.Node.Release();
                            }
                        }
                    }
                }
            }

            foreach (var layer in graph.Outputs)
                InferLayer(layer);

            return context;
        }

        private static void AllocateInputMemoryDefault(Layer layer, OutputConnector input, InferenceContext context)
        {
            switch (layer)
            {
                case K210Conv2d _:
                    context.GetOrAllocateKPUMemory(input);
                    break;
                default:
                    context.GetOrAllocateMainMemory(input);
                    break;
            }
        }
    }
}
