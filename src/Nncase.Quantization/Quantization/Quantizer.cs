﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
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

    public Quantizer(IEGraph graph, QuantizeOptions quantizeOptions)
    {
        _graph = graph;
        _quantizeOptions = quantizeOptions;
        MarkRangeOfs();
    }

    public async Task RunAsync(RunPassContext options)
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

    /// <summary>
    /// collec all rangeof enode.
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
}
