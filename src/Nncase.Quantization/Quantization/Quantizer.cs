// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using DryIoc;
using Microsoft.Extensions.Options;
using Newtonsoft.Json;
using Nncase.Diagnostics;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.Passes;
using Nncase.PatternMatch;
using Nncase.Utilities;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Quantization;
internal partial class Quantizer
{
    private readonly IEGraph _graph;
    private readonly Expr _expr;
    private readonly QuantizeOptions _quantizeOptions;
    private readonly Dictionary<Var, QuantConfig>? _fakeNodeConfigs = new();

    public Quantizer(IEGraph graph, QuantizeOptions quantizeOptions)
    {
        _graph = graph;
        _expr = ExtractExpr();
        _quantizeOptions = quantizeOptions;
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
        var ranges = await GetRangesAsync(_quantizeOptions.CalibrationDataset!);

        // ranges = ranges.ToDictionary(item => item.Key, item => FixUpRange(item.Value));
        ranges = ranges.ToDictionary(item => item.Key, item => item.Value.Select(v => FixUpRange(v)).ToArray());

        if (_quantizeOptions.CalibrationMethod is CalibMethod.Kld)
        {
            // 1.1. Get histograms
            var histograms = await GetHistogramsAsync(_quantizeOptions.CalibrationDataset, ranges, srcBinSize, dstBinSize);

            // 1.2. Select best ranges
            ranges = GetOptRanges(histograms, ranges, srcBinSize, dstBinSize, _quantizeOptions.CalibrationMethod);
        }

        UpdateFakeNodesWithRange(ranges);
        AssignRanges();
    }

    private Expr ExtractExpr()
    {
        Func<ENode, bool> fakeCallFilter = node =>
        {
            return node.Expr is Call call &&
                            (call.Target is QuantizeOp);
        };

        Func<ENode, Tuple<ENode, ENode>> replaceUninitlizedWithVar = node =>
        {
            var parameters = ((Call)node.Expr).Arguments.ToArray();
            parameters[^1] = new Var("placeholder", new TensorType(DataTypes.Float32, Shape.Unknown(1)));
            var newCall = ((Call)node.Expr).With(arguments: parameters);
            var newEnode = ENode.Create(newCall, Array.Empty<EClass>());
            _fakeNodeConfigs?.Add((Var)parameters[^1], default!); // config detail info will be update later.
            return new Tuple<ENode, ENode>(node, newEnode);
        };

        var callPairs = _graph.Nodes.Where(node => fakeCallFilter(node)).Select(node => replaceUninitlizedWithVar(node)).ToArray();

        foreach ((var nodeWithUninitlized, var nodeWithVar) in callPairs)
        {
            var rangeEclass = _graph.Add(nodeWithVar.Expr);
            var rangeOfEclass = _graph.Find(nodeWithUninitlized);
            _graph.Union(rangeOfEclass, rangeEclass);
        }

        _graph.Rebuild();

        return _graph.Extract(_graph.Root!, null, out _);
    }

    private void AssignRanges()
    {
        if (_fakeNodeConfigs is not null)
        {
            foreach (var (var, config) in _fakeNodeConfigs)
            {
                if (config.IsEmpty())
                {
                    continue;
                }

                var eNode = ENode.Create(var, Array.Empty<EClass>());
                var rangeEclass = _graph.Add(config.ToRaw());
                var rangeOfEclass = _graph.Find(eNode);
                _graph.Union(rangeOfEclass, rangeEclass);
            }

            _graph.Rebuild();

            // var debugExpr = _graph.Extract(_graph.Root!, null, out _);
        }
    }

    private void UpdateFakeNodesWithRange(IDictionary<Expr, ValueRange<float>[]> ranges)
    {
        float[] argRange = new float[2];
        foreach (var (key, range) in ranges)
        {
            if (key is Call &&
                (((Call)key).Target is QuantizeOp))
            {
                // 输入信息的填充
                var parameters = ((Op)((Call)key).Target).Parameters.ToArray();
                var configHeader = new float[] { parameters.Length, range.Length };
                var inputConfigs = new List<QuantConfigData>();
                var outConfigs = new List<QuantConfigData>();
                for (int argIdx = 0; argIdx < ((Call)key).Arguments.Length; argIdx++)
                {
                    var arg = ((Call)key).Arguments[argIdx];
                    if (arg is not Nncase.IR.None)
                    {
                        if (parameters[argIdx].ParameterKind == ParameterKind.Weights)
                        {
                            var dType = DataTypes.UInt8;
                            var weights = (TensorConst)((Call)key).Arguments[argIdx];
                            var weightsValue = weights.Value.ToArray<float>();
                            var oc = weights.CheckedShape[0].FixedValue;
                            var minMaxArr = QuantUtility.GetWeightsRangesByChannel(weightsValue, oc);
                            inputConfigs.Add(new QuantConfigData(Tensor.From(minMaxArr.ToArray(), new[] { oc, 2 }), dType));
                        }
                        else
                        {
                            // 每个parameter的range只有一个，所以这里固定索引为0
                            var dType = DataTypes.UInt8;
                            argRange = new float[] { ranges[arg][0].Min, ranges[arg][0].Max };
                            inputConfigs.Add(new QuantConfigData(new Tensor<float>(argRange, new[] { 1, 2 }), dType));
                        }
                    }
                    else
                    {
                        var dType = DataTypes.UInt8;
                        argRange = new float[] { float.MinValue, float.MaxValue };
                        inputConfigs.Add(new QuantConfigData(new Tensor<float>(argRange, new[] { 1, 2 }), dType));
                    }
                }

                // 输出range
                foreach (var outRange in range)
                {
                    var dType = DataTypes.UInt8;
                    argRange = new float[] { outRange.Min, outRange.Max };
                    outConfigs.Add(new QuantConfigData(new Tensor<float>(argRange, new[] { 1, 2 }), dType));
                }

                var quantInfo = configHeader.Concat(inputConfigs.SelectMany(x => x.ToRaw())).Concat(outConfigs.SelectMany(x => x.ToRaw())).ToArray();
                var quantConfig = QuantConfig.FromRaw(quantInfo);
                _fakeNodeConfigs![(Var)((Call)key).Arguments[^1]] = quantConfig;
            }
        }
    }

    private async Task<IDictionary<Expr, ValueRange<float>[]>> GetRangesAsync(ICalibrationDatasetProvider calibrationDataset)
    {
        var ranges = new Dictionary<Expr, ValueRange<float>[]>(ReferenceEqualityComparer.Instance);
        await RunPassAsync(calibrationDataset, (values) =>
        {
            foreach (var value in values)
            {
                int globalVarIndex = 0;
                if (value.Key is Call && ((Call)value.Key).Arguments[^1] is Var)
                {
                    globalVarIndex = ((Var)((Call)value.Key).Arguments[^1]).GlobalVarIndex;
                }

                try
                {
                    var outputNum = value.Value.AsTensors().Length;
                    var outRange = new ValueRange<float>[outputNum];
                    for (int i = 0; i < outputNum; i++)
                    {
                        var tensor = Tensor.From(value.Value.AsTensors()[i].ToArray<float>());
                        var range = GetMinMax(tensor);
                        if (ranges.TryGetValue(value.Key, out var oldRange))
                        {
                            ranges[value.Key][i] = oldRange[i].Union(range);
                        }
                        else
                        {
                            outRange[i] = range;
                            ranges.Add(value.Key, outRange);
                        }
                    }
                }
                catch (Exception)
                {
                    continue;
                }
            }
        });
        return ranges;
    }

    private Dictionary<Expr, IValue> GetMemo(Dictionary<Var, IValue> sample)
    {
        if (_fakeNodeConfigs is not null && _fakeNodeConfigs.Count != 0)
        {
            foreach (var (var, _) in _fakeNodeConfigs!)
            {
                sample.Add(var, IR.F.Random.Normal(DataTypes.Float32, new int[] { 1 }).Evaluate());
            }
        }

        return EvaluatorUtil.GetMemo(_expr, sample);
    }

    private async Task RunPassAsync(ICalibrationDatasetProvider calibrationDataset, Action<IReadOnlyDictionary<Expr, IValue>> func)
    {
        await foreach (Dictionary<Var, IValue> sample in calibrationDataset.Samples)
        {
            var memo = GetMemo(sample);
            func(memo);
            GC.Collect();
        }
    }

    private async Task<IDictionary<Expr, QuantizeHistogram<float>[]>> GetHistogramsAsync(ICalibrationDatasetProvider calibrationDataset, IDictionary<Expr, ValueRange<float>[]> ranges, int srcBinSize, int dstBinSize)
    {
        var histograms = new Dictionary<Expr, QuantizeHistogram<float>[]>(ReferenceEqualityComparer.Instance);
        histograms = ranges.ToDictionary(range => range.Key, range => Enumerable.Range(0, range.Value.Length)
                       .Select(i => new QuantizeHistogram<float>(
                           new List<float>(new float[srcBinSize]),
                           new List<float>(new float[srcBinSize]))).ToArray());

        await RunForHistogramsAsync(calibrationDataset, ranges, childrenValues =>
        {
            foreach (var (key, value) in childrenValues)
            {
                for (int i = 0; i < value.Length; i++)
                {
                    var r = ranges[key][i].Max - ranges[key][i].Min;
                    var srcBinInterval = r / srcBinSize;

                    var childrenTensor = value[i].Cast<float>();
                    var childrenBuffer = childrenTensor.Buffer.Span;
                    var valueRange = ranges[key][i];
                    var histogram = histograms[key][i];
                    foreach (var buf in childrenBuffer)
                    {
                        var r_index = (buf - valueRange.Min) / srcBinInterval;
                        var index = (int)Math.Clamp((float)r_index, 0F, (float)srcBinSize - 1);
                        histogram.SrcBin[index]++;
                    }
                }
            }
        });
        return histograms;
    }

    private async Task RunForHistogramsAsync(ICalibrationDatasetProvider calibrationDataset, IDictionary<Expr, ValueRange<float>[]> ranges, Action<IReadOnlyDictionary<Expr, Tensor[]>> func)
    {
        await foreach (Dictionary<Var, IValue> sample in calibrationDataset.Samples)
        {
            var memoTensors = new Dictionary<Expr, Tensor[]>();
            var memo = GetMemo(sample);
            foreach (var (key, _) in ranges)
            {
                try
                {
                    memoTensors[key] = memo[key].AsTensors();
                }
                catch (Exception)
                {
                    continue;
                }
            }

            func(memoTensors);
            GC.Collect();
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
            if (r < 1e-6f)
            {
                r = 0.1f;
            }

            range.Max = range.Min + r;
        }

        return range;
    }

    private IDictionary<Expr, ValueRange<float>[]> GetOptRanges(IDictionary<Expr, QuantizeHistogram<float>[]> histograms, IDictionary<Expr, ValueRange<float>[]> ranges, int srcBinSize, int dstBinSize, CalibMethod calibrationMethod)
    {
        var optRanges = new Dictionary<Expr, ValueRange<float>[]>(ReferenceEqualityComparer.Instance);
        if (calibrationMethod == CalibMethod.Kld)
        {
            foreach (KeyValuePair<Expr, QuantizeHistogram<float>[]> histogram in histograms)
            {
                var optRange = new ValueRange<float>[histogram.Value.Length];
                for (int i = 0; i < histogram.Value.Length; i++)
                {
                    histogram.Value[i].SrcBin = Smooth(histogram.Value[i].SrcBin);
                    var minKld = float.MaxValue;
                    var r = ranges[histogram.Key][i].Max - ranges[histogram.Key][i].Min;
                    var srcBinInterval = r / srcBinSize;
                    var betterThreshold = new Tuple<int, int>(0, srcBinSize);
                    var zeroThreshold = (int)Math.Clamp((0 - ranges[histogram.Key][i].Min) / srcBinInterval, 0, srcBinSize - 1);

                    // range max first
                    int lowerThreshold = 0;
                    var srcBin = histogram.Value[i].SrcBin;
                    for (int upperThreshold = srcBinSize; upperThreshold >= dstBinSize && upperThreshold >= zeroThreshold; upperThreshold -= dstBinSize)
                    {
                        GetKldOptRanges(lowerThreshold, upperThreshold, dstBinSize, ref srcBin, ref minKld, ref betterThreshold);
                    }

                    // range min
                    minKld = float.MaxValue;
                    int upperThreshold2 = betterThreshold.Item2;
                    for (int lowerThreshold2 = 0; lowerThreshold2 <= zeroThreshold && lowerThreshold2 <= upperThreshold2 - dstBinSize; lowerThreshold2 += dstBinSize)
                    {
                        GetKldOptRanges(lowerThreshold2, upperThreshold2, dstBinSize, ref srcBin, ref minKld, ref betterThreshold);
                    }

                    var optMin = ((betterThreshold.Item1 - 0.5f) * srcBinInterval) + ranges[histogram.Key][i].Min;
                    var optMax = ((betterThreshold.Item2 + 0.5f) * srcBinInterval) + ranges[histogram.Key][i].Min;
                    optRange[i] = new ValueRange<float>(optMin, optMax);
                }

                optRanges.Add(histogram.Key, optRange);
            }

            return optRanges;
        }
        else
        {
            throw new ArgumentException("Invalid calibration method.");
        }
    }
}
