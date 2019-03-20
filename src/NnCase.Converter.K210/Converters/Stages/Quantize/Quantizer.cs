using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using NnCase.Converter.Data;
using NnCase.Converter.Model;
using NnCase.Converter.Model.Layers;
using TensorFlow;
using NnCase.Converter.K210.Converters.Layers;

#if NET471
using System.Collections.Async;
#endif

namespace NnCase.Converter.K210.Converters.Stages.Quantize
{
    public static class Quantizer
    {
        public static async Task<QuantizationContext> QuantizeAsync(Dataset dataset, GraphPlanContext planContext)
        {
            using (var session = new TFSession(planContext.TFGraph))
            {
                var connectors = new List<OutputConnector>();
                var additionalOutputs = new List<Guid>();
                var toFetches = new List<TFOutput>();

                foreach (var output in planContext.TFOutputs)
                {
                    connectors.Add(output.Key);
                    if (!(output.Key.Owner is InputLayer))
                        toFetches.Add(output.Value);
                }

                foreach (var additional in planContext.AdditionalTFOutputs)
                {
                    additionalOutputs.Add(additional.Key);
                    toFetches.Add(additional.Value);
                }

                var quantizationContext = new QuantizationContext
                {
                    Outputs = connectors,
                    AdditionalOutputs = additionalOutputs,
                    PlanContext = planContext,
                    DatasetProcess = dataset.PostprocessMethod
                };

#if NET471
                await dataset.GetBatchesAsync().ForEachAsync(async batch =>
#else
                await foreach (var batch in dataset.GetBatchesAsync())
#endif
                {
                    var input = batch.ToNHWC();
                    var runner = session.GetRunner();

                    runner.AddInput(planContext.Inputs.Values.First(), input);
                    foreach (var fetch in toFetches)
                        runner.Fetch(fetch);

                    var outputs = runner.Run();
                    RecordOutputs(new[] { input }.Concat(outputs).ToList(), quantizationContext);
                }
#if NET471
                );
#endif

                var converters = (from t in typeof(Quantizer).Assembly.ExportedTypes
                                  let attrs = t.GetCustomAttributes<LayerConverterAttribute>()
                                  where attrs.Any()
                                  from attr in attrs
                                  select new
                                  {
                                      Key = attr.Type,
                                      Value = new { Type = t, Method = t.GetMethod("FixupQuantization") }
                                  }).ToDictionary(x => x.Key, x => x.Value);
                foreach (var layer in planContext.TFOutputs.Keys.Select(x => x.Owner).Distinct())
                {
                    if (converters.TryGetValue(layer.GetType(), out var info) && info.Method != null)
                    {
                        var converter = Activator.CreateInstance(info.Type);
                        info.Method.Invoke(converter, new object[] { layer, quantizationContext });
                    }
                }

                return quantizationContext;
            }
        }

        private static unsafe void RecordOutputs(IReadOnlyList<TFTensor> outputs, QuantizationContext context)
        {
            for (int i = 0; i < outputs.Count; i++)
            {
                var span = new Span<float>(outputs[i].Data.ToPointer(), (int)outputs[i].TensorByteSize / 4);
                var newRange = GetClippedRange(span);

                if (i < context.Outputs.Count)
                {
                    var conn = context.Outputs[i];
                    if (context.Distributions.TryGetValue(conn, out var range))
                        context.Distributions[conn] = range.EMA(0.01, newRange);
                    else
                        context.Distributions.Add(conn, newRange);
                }
                else
                {
                    var idx = i - context.Outputs.Count;
                    var conn = context.AdditionalOutputs[idx];
                    if (context.AdditionalDistributions.TryGetValue(conn, out var range))
                        context.AdditionalDistributions[conn] = range.EMA(0.01, newRange);
                    else
                        context.AdditionalDistributions.Add(conn, newRange);
                }
            }
        }

        public static QuantizationRange GetRange(ReadOnlySpan<float> data)
        {
            double min = double.MaxValue, max = double.MinValue;
            bool used = false;
            for (int j = 0; j < data.Length; j++)
            {
                if (Math.Abs(data[j]) > 100) continue;
                used = true;
                min = Math.Min(min, data[j]);
                max = Math.Max(max, data[j]);
            }

            if (!used || Math.Abs(min) > 100 || Math.Abs(max) > 100)
                return QuantizationRange.Default;
            else if (min == max)
                return new QuantizationRange { Min = -1, Max = 1 };
            else
                return new QuantizationRange { Min = min, Max = max };
        }

        public static QuantizationRange GetClippedRange(ReadOnlySpan<float> data) =>
            Clipper.GetClippedRange(data);

        public static double Quantize(double value, double scale, double bias)
        {
            return value * scale - bias;
        }

        public static double Quantize(ReadOnlySpan<float> data, Span<ushort> dest, double scale, double bias, int weightsBits)
        {
            ushort max = (ushort)((1 << weightsBits) - 1);

            for (int i = 0; i < data.Length; i++)
                dest[i] = (ushort)
#if NET471
                    FxExtensions
#else
                    Math
#endif
                    .Clamp(Math.Round(data[i] * scale - bias), 0, max);

            var diff = new double[data.Length];
            for (int i = 0; i < data.Length; i++)
                diff[i] = Math.Abs(((dest[i] + bias) / scale) - data[i]);
            var avg = diff.Max();
            return avg;
        }

        public static ushort[] Quantize(ReadOnlySpan<float> data, double scale, double bias, int weightsBits)
        {
            var q = new ushort[data.Length];
            Quantize(data, q, scale, bias, weightsBits);
            return q;
        }

        public static (double value, int shift) ExtractValueAndShift(double value, int maxBits, int maxShift)
        {
            int shift = 0;
            double mul = 0;

            if (Math.Abs(value) > 1)
            {
                mul = C.math.frexp(value, out var mulShift);
                shift = Math.Min(maxShift, maxBits - 1 - mulShift);
                mul = mul * Math.Pow(2, shift + mulShift);
            }
            else if (value == 0)
            {
                mul = shift = 0;
            }
            else
            {
                mul = C.math.frexp(value, out var mulShift);
                shift = Math.Min(maxShift + mulShift, maxBits - 1);
                mul = mul * Math.Pow(2, shift);
                shift -= mulShift;
            }

            Debug.Assert(Math.Abs(mul) < Math.Pow(2, maxBits - 1));
            Debug.Assert(shift <= maxShift);
            Debug.Assert(Math.Abs(value - mul * Math.Pow(2, -shift)) <= double.Epsilon);
            return (mul, shift);
        }

        public static byte[] GetRequantizeTable(QuantizationRange inputRange, QuantizationRange outputRange)
        {
            (var si, var bi) = inputRange.GetScaleBias(8);
            (var so, var bo) = outputRange.GetScaleBias(8);
            var s = so / si;

            var table = new byte[256];
            for (int i = 0; i < 256; i++)
                table[i] = (byte)FxExtensions.Clamp(Math.Round((i + bi) * s - bo), 0, 255);
            return table;
        }
    }
}
