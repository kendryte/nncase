// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.Passes;
using Nncase.PatternMatch;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Quantization;

internal partial class Quantizer
{
    private readonly IEGraph _graph;
    private readonly QuantizeOptions _quantizeOptions;
    private readonly List<ENode> _rangeOfs = new List<ENode>();
    private readonly List<ENode> _childrenOfRangeOfs = new List<ENode>();
    private readonly List<ENode> _markers = new List<ENode>();

    public Quantizer(IEGraph graph, QuantizeOptions quantizeOptions)
    {
        _graph = graph;
        _quantizeOptions = quantizeOptions;
        MarkRangeOfs();
        MarkMarkers();
    }

    public async Task RunAsync(RunPassContext options)
    {
        bool configExist = _quantizeOptions.QuantScheme != string.Empty;
        bool exportQuantScheme = _quantizeOptions.ExportQuantScheme;
        bool exportWeightRangeByChannel = _quantizeOptions.ExportWeightRangeByChannel;

        if (!configExist)
        {
            int srcBinSize = 8192;
            int dstBinSize = 256;

            if (_quantizeOptions.CalibrationDataset == null)
            {
                throw new InvalidOperationException($"{nameof(_quantizeOptions.CalibrationDataset)} is not set");
            }

            // 1.0 Get ranges
            var ranges = await GetRangesAsync(_quantizeOptions.CalibrationDataset);

            ranges = ranges.ToDictionary(item => item.Key, item => FixUpRange(item.Value));

            if (_quantizeOptions.CalibrationMethod is CalibMethod.Kld)
            {
                // 1.1. Get histograms
                var histograms = await GetHistogramsAsync(_quantizeOptions.CalibrationDataset, ranges, srcBinSize, dstBinSize);

                // 1.2. Select best ranges
                var optRanges = GetOptRanges(histograms, ranges, srcBinSize, dstBinSize, _quantizeOptions.CalibrationMethod);

                // 1.3. Assign ranges
                AssignRanges(optRanges);
            }
            else
            { // 2. Assign ranges
                AssignRanges(ranges);
            }

            // 3. Export quant info
            if (_quantizeOptions.ExportQuantScheme == true)
            {
                var quantScheme = new QuantScheme();
                quantScheme.Version = "1.0";
                var outputsCount = 0;
                foreach (var range in ranges)
                {
                    if (range.Key.Expr.Metadata.OutputNames != null)
                    {
                        outputsCount++;
                    }
                }

                quantScheme.Outputs = new Output[outputsCount];

                if (_quantizeOptions.ExportWeightRangeByChannel == false)
                {
                    var index = 0;
                    foreach (var range in ranges.Where(r => r.Key.Expr.Metadata.OutputNames != null))
                    {
                        quantScheme.Outputs[index] = new Output();
                        quantScheme.Outputs[index].Name = range.Key.Expr.Metadata.OutputNames![0];
                        quantScheme.Outputs[index].DataRangeMode = "by_tensor";
                        quantScheme.Outputs[index].DataType = ((RangeOf)((Call)range.Key.Expr).Target).IsRangeOfWeight == true ? _quantizeOptions.WQuantType.ToString() : _quantizeOptions.QuantType.ToString();
                        quantScheme.Outputs[index].DataRange = new ValueRange<float>[1];
                        quantScheme.Outputs[index].DataRange![0] = range.Value;
                        index++;
                    }
                }
                else
                {
                    var weightsByChannelRanges = new Dictionary<ENode, ValueRange<float>[]>(ReferenceEqualityComparer.Instance);
                    foreach (var range in ranges)
                    {
                        if (((RangeOf)((Call)range.Key.Expr).Target).IsRangeOfWeight == true)
                        {
                            var oc = range.Key.Children[1].Nodes[0].Expr.CheckedShape[0].FixedValue;
                            var valueRanges = new ValueRange<float>[oc];
                            var weightsValue = ((TensorConst)range.Key.Children[1].Nodes[0].Expr).Value.ToArray<float>();
                            var weightsSize = weightsValue.Length;
                            var eachChannelSize = weightsSize / oc;
                            var tmpMin = float.MaxValue;
                            var tmpMax = float.MinValue;

                            for (int i = 0; i < weightsSize; i++)
                            {
                                if (weightsValue[i] < tmpMin)
                                {
                                    tmpMin = weightsValue[i];
                                }

                                if (weightsValue[i] > tmpMax)
                                {
                                    tmpMax = weightsValue[i];
                                }

                                if ((i + 1) % eachChannelSize == 0)
                                {
                                    valueRanges[i / eachChannelSize] = new ValueRange<float> { Min = tmpMin, Max = tmpMax };
                                    tmpMin = float.MaxValue;
                                    tmpMax = float.MinValue;
                                }
                            }

                            weightsByChannelRanges.Add(range.Key, valueRanges);
                        }
                        else
                        {
                            var valueRanges = new ValueRange<float>[1];
                            valueRanges[0] = range.Value;
                            weightsByChannelRanges.Add(range.Key, valueRanges);
                        }
                    }

                    var index = 0;
                    foreach (var range in ranges.Where(r => r.Key.Expr.Metadata.OutputNames != null))
                    {
                        quantScheme.Outputs[index] = new Output();
                        quantScheme.Outputs[index].Name = range.Key.Expr.Metadata.OutputNames![0];
                        quantScheme.Outputs[index].DataRangeMode = ((RangeOf)((Call)range.Key.Expr).Target).IsRangeOfWeight == true ? "by_channel" : "by_tensor";
                        quantScheme.Outputs[index].DataType = ((RangeOf)((Call)range.Key.Expr).Target).IsRangeOfWeight == true ? _quantizeOptions.WQuantType.ToString() : _quantizeOptions.QuantType.ToString();
                        var rangeLength = weightsByChannelRanges[range.Key].Length;
                        quantScheme.Outputs[index].DataRange = new ValueRange<float>[rangeLength];
                        for (int i = 0; i < rangeLength; i++)
                        {
                            quantScheme.Outputs[index].DataRange![i] = weightsByChannelRanges[range.Key][i];
                        }

                        index++;
                    }
                }

                var quantSchemeString = JsonConvert.SerializeObject(quantScheme);
                _quantizeOptions.QuantScheme = quantSchemeString;
            }
        }
        else
        {
            string readJson = _quantizeOptions.QuantScheme;
            var quantScheme = JsonConvert.DeserializeObject<QuantScheme>(readJson);
            var ranges = GetRangesFromConfig(quantScheme!);
            AssignByChannelRanges(ranges);
            AssignDataTypeFromConfig(quantScheme!);
        }

        // // 3. Choose better quant method using cosine, and bind info with ir.
        // if (quantOptions.BindQuantMethod)
        // {
        //     var info = await options.Target.BindQuantMethodCosine(quantOptions.CalibrationDataset, options.Target, _rangeOfs, _childrenOfRangeOfs, _context);
        // }
        _graph.Rebuild();
    }

    private async Task RunPassAsync(ICalibrationDatasetProvider calibrationDataset, Action<IReadOnlyDictionary<ENode, Tensor>, IReadOnlyDictionary<ENode, Tensor>> func)
    {
        await foreach (var sample in calibrationDataset.Samples)
        {
            IReadOnlyDictionary<ENode, Tensor> values, childrenValues;
            using (var dumpScope = new DumpScope("ep1"))
            {
                var evaluator = new CalibrationEvaluator(sample, _rangeOfs);
                values = evaluator.Evaluate();
            }

            using (var dumpScope2 = new DumpScope("ep2"))
            {
                var childrenEvaluator = new CalibrationEvaluator(sample, _childrenOfRangeOfs);
                childrenValues = childrenEvaluator.Evaluate();
            }

            // values are children op range values(only two scalars for each value: Min and Max), childrenValues are children op tensor values.
            func(values, childrenValues);
        }
    }

    private async Task RunPassAsync(ICalibrationDatasetProvider calibrationDataset, Action<IReadOnlyDictionary<ENode, Tensor>> func)
    {
        await foreach (var sample in calibrationDataset.Samples)
        {
            using var dumpScope = new DumpScope("ep1");
            var evaluator = new CalibrationEvaluator(sample, _rangeOfs);
            var values = evaluator.Evaluate();
            func(values);
        }
    }

    private ValueRange<float> FixUpRange(ValueRange<float> range, bool symmetric = false)
    {
        if (symmetric)
        {
            var r = Math.Max(0.01f, Math.Max(Math.Abs(range.Min), Math.Abs(range.Max)));
            return new ValueRange<float>(-r, r);
        }
        else
        {
            if (range.Max < 0)
            {
                range.Max = 0;
            }

            if (range.Min > 0)
            {
                range.Min = 0;
            }

            var r = range.Max - range.Min;
            if (r == 0)
            {
                r = 0.1f;
            }

            range.Max = range.Min + r;
        }

        return range;
    }

    private async Task<IDictionary<ENode, ValueRange<float>>> GetRangesAsync(ICalibrationDatasetProvider calibrationDataset)
    {
        var ranges = new Dictionary<ENode, ValueRange<float>>(ReferenceEqualityComparer.Instance);
        await RunPassAsync(calibrationDataset, (values) =>
        {
            foreach (var value in values)
            {
                var tensor = value.Value.Cast<float>();
                var range = GetMinMax(tensor);
                if (ranges.TryGetValue(value.Key, out var oldRange))
                {
                    ranges[value.Key] = oldRange.Union(range);
                }
                else
                {
                    ranges.Add(value.Key, range);
                }
            }
        });
        return ranges;
    }

    private IDictionary<ENode, ValueRange<float>[]> GetRangesFromConfig(QuantScheme quantScheme)
    {
        var ranges = new Dictionary<ENode, ValueRange<float>[]>(ReferenceEqualityComparer.Instance);

        foreach (var rangeOf in _rangeOfs)
        {
            for (int i = 0; i < quantScheme!.Outputs!.Length; i++)
            {
                if (rangeOf.Expr.Metadata.OutputNames?[0] == quantScheme!.Outputs[i].Name)
                {
                    if (((RangeOf)((Call)rangeOf.Expr).Target).IsRangeOfWeight == true && quantScheme!.Outputs[i].DataRangeMode == "by_tensor")
                    {
                        var oc = ((Call)rangeOf.Expr).Operands[1].CheckedShape[0].FixedValue;
                        var valueRanges = new ValueRange<float>[oc];
                        for (int j = 0; j < oc; j++)
                        {
                            valueRanges[j] = FixUpRange(new ValueRange<float>((float)quantScheme!.Outputs[i].DataRange![0].Min, (float)quantScheme!.Outputs[i].DataRange![0].Max));
                        }

                        ranges.Add(rangeOf, valueRanges);
                    }
                    else
                    {
                        var valueRanges = new ValueRange<float>[quantScheme!.Outputs[i].DataRange!.Length];
                        for (int j = 0; j < quantScheme!.Outputs[i].DataRange!.Length; j++)
                        {
                            valueRanges[j] = FixUpRange(new ValueRange<float>((float)quantScheme!.Outputs[i].DataRange![j].Min, (float)quantScheme!.Outputs[i].DataRange![j].Max));
                        }

                        ranges.Add(rangeOf, valueRanges);
                    }
                }
            }
        }

        return ranges;
    }

    private void AssignDataTypeFromConfig(QuantScheme quantScheme)
    {
        foreach (var marker in _markers)
        {
            for (int i = 0; i < quantScheme!.Outputs!.Length; i++)
            {
                if (marker.Expr.Metadata.OutputNames?[0] == quantScheme.Outputs[i].Name)
                {
                    var markerExpr = (Marker)marker.Expr;
                    if (markerExpr.MixQuantInfo == null)
                    {
                        markerExpr.MixQuantInfo = new MixQuantInfo();
                    }

                    DataType dataType = DataTypes.FromShortName(quantScheme!.Outputs[i].DataType!);
                    markerExpr.MixQuantInfo!.MarkerQuantType = dataType;
                }
            }
        }
    }

    private async Task<IDictionary<ENode, QuantizeHistogram<float>>> GetHistogramsAsync(ICalibrationDatasetProvider calibrationDataset, IDictionary<ENode, ValueRange<float>> ranges, int srcBinSize, int dstBinSize)
    {
        var histograms = new Dictionary<ENode, QuantizeHistogram<float>>(ReferenceEqualityComparer.Instance);
        await RunPassAsync(calibrationDataset, (values, childrenValues) =>
        {
            var valuesList = values.ToList();
            var childrenValuesList = childrenValues.ToList();
            for (int i = 0; i < valuesList.Count; i++)
            {
                var r = ranges[valuesList[i].Key].Max - ranges[valuesList[i].Key].Min;
                var srcBinInterval = r / srcBinSize;
                if (!histograms.TryGetValue(valuesList[i].Key, out var histogram))
                {
                    var initSrcBin = new List<float>(new float[srcBinSize]);
                    var initDstBin = new List<float>(new float[dstBinSize]);
                    histogram = new QuantizeHistogram<float>(initSrcBin, initDstBin);
                    histograms.Add(valuesList[i].Key, histogram);
                }

                var childrenTensor = childrenValuesList[i].Value.Cast<float>();
                var childrenBuffer = childrenTensor.Buffer.Span;
                var valueRange = ranges[valuesList[i].Key];

                foreach (var buf in childrenBuffer)
                {
                    var r_index = (buf - valueRange.Min) / srcBinInterval;
                    var index = (int)Math.Clamp((float)r_index, 0F, (float)srcBinSize - 1);
                    histogram.SrcBin[index]++;
                }
            }
        });
        return histograms;
    }

    private IDictionary<ENode, ValueRange<float>> GetOptRanges(IDictionary<ENode, QuantizeHistogram<float>> histograms, IDictionary<ENode, ValueRange<float>> ranges, int srcBinSize, int dstBinSize, CalibMethod calibrationMethod)
    {
        var optRanges = new Dictionary<ENode, ValueRange<float>>(ReferenceEqualityComparer.Instance);
        if (calibrationMethod == CalibMethod.Kld)
        {
            foreach (KeyValuePair<ENode, QuantizeHistogram<float>> histogram in histograms)
            {
                histogram.Value.SrcBin = Smooth(histogram.Value.SrcBin);
                var minKld = float.MaxValue;
                var r = ranges[histogram.Key].Max - ranges[histogram.Key].Min;
                var srcBinInterval = r / srcBinSize;
                var betterThreshold = new Tuple<int, int>(0, srcBinSize);
                var zeroThreshold = (int)Math.Clamp((0 - ranges[histogram.Key].Min) / srcBinInterval, 0, srcBinSize - 1);

                // range max first
                int lowerThreshold = 0;
                var srcBin = histogram.Value.SrcBin;
                for (int upperThreshold = srcBinSize; upperThreshold >= dstBinSize && upperThreshold >= zeroThreshold; upperThreshold -= dstBinSize)
                {
                    GetKldOptRanges(lowerThreshold, upperThreshold, dstBinSize, ref srcBin, ref minKld, ref betterThreshold);

                    // betterThreshold = thresholdWithMinKldWithSmoothSrcBin.Item1;
                    // minKld = thresholdWithMinKldWithSmoothSrcBin.Item2;
                    // srcBin = thresholdWithMinKldWithSmoothSrcBin.Item3;
                }

                // range min
                minKld = float.MaxValue;
                int upperThreshold2 = betterThreshold.Item2;
                for (int lowerThreshold2 = 0; lowerThreshold2 <= zeroThreshold && lowerThreshold2 <= upperThreshold2 - dstBinSize; lowerThreshold2 += dstBinSize)
                {
                    GetKldOptRanges(lowerThreshold2, upperThreshold2, dstBinSize, ref srcBin, ref minKld, ref betterThreshold);

                    // betterThreshold = thresholdWithMinKldWithSmoothSrcBin.Item1;
                    // minKld = thresholdWithMinKldWithSmoothSrcBin.Item2;
                    // srcBin = thresholdWithMinKldWithSmoothSrcBin.Item3;
                }

                var optMin = ((betterThreshold.Item1 - 0.5f) * srcBinInterval) + ranges[histogram.Key].Min;
                var optMax = ((betterThreshold.Item2 + 0.5f) * srcBinInterval) + ranges[histogram.Key].Min;
                optRanges.Add(histogram.Key, new ValueRange<float>(optMin, optMax));
            }

            return optRanges;
        }
        else
        {
            throw new ArgumentException("Invalid calibration method.");
        }
    }

    private void AssignRanges(IDictionary<ENode, ValueRange<float>> ranges)
    {
        // note union the constant in the rangeof eclass, when extact the graph will replace the rangeof expression with the constant ValueRange.
        foreach (var range in ranges)
        {
            var value = new[] { range.Value.Min, range.Value.Max };
            var rangeEclass = _graph.Add(value);
            var rangeOfEclass = _graph.Find(range.Key);
            _graph.Union(rangeOfEclass, rangeEclass);
        }

        // _graph.Rebuild();
    }

    private void AssignByChannelRanges(IDictionary<ENode, ValueRange<float>[]> ranges)
    {
        // note union the constant in the rangeof eclass, when extact the graph will replace the rangeof expression with the constant ValueRange.
        foreach (var range in ranges)
        {
            var value = range.Value;
            var oc = value.Length;
            var minMaxArrSize = oc * 2;
            var minMaxArr = new float[minMaxArrSize];
            for (int i = 0; i < minMaxArrSize; i++)
            {
                minMaxArr[i] = i % 2 == 0 ? value[i / 2].Min : value[i / 2].Max;
            }

            var shape = oc == 1 ? new[] { 2 } : new[] { oc, 2 };
            var rangeEclass = _graph.Add(new TensorConst(Tensor.From(minMaxArr, shape)));
            var rangeOfEclass = _graph.Find(range.Key);
            range.Key.Expr.CheckedType = rangeEclass.CheckedType;
            rangeOfEclass.SetCheckedType(rangeEclass.CheckedType);
            _graph.Union(rangeOfEclass, rangeEclass);
        }
    }

    /// <summary>
    /// collect all rangeof enode.
    /// </summary>
    private void MarkRangeOfs()
    {
        if (EGraphMatcher.TryMatchRoot(_graph.Nodes, IsRangeOf(IsWildcard()), out var matches))
        {
            foreach (var match in matches)
            {
                var rangeOf = (ENode)match.Root;
                _rangeOfs.Add(rangeOf);
                _childrenOfRangeOfs.Add(rangeOf.Children[1].Nodes[0]);
            }
        }
    }

    /// <summary>
    /// collect all marker enode.
    /// </summary>
    private void MarkMarkers()
    {
        foreach (var node in _graph.Nodes)
        {
            if (node.Expr is Marker)
            {
                _markers.Add(node);
            }
        }
    }
}
