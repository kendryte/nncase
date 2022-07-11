// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.PatternMatch;
using Nncase.Transform;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Quantization;

internal partial class Quantizer
{
    private readonly EGraph _graph;
    private readonly List<ENode> _rangeOfs = new List<ENode>();

    public Quantizer(EGraph graph)
    {
        _graph = graph;
        MarkRangeOfs();
    }

    public async Task RunAsync(QuantizeOptions options)
    {
        if (options.CalibrationDataset == null)
        {
            throw new ArgumentNullException(nameof(options.CalibrationDataset));
        }

        // 1. Get ranges
        var ranges = await GetRangesAsync(options.CalibrationDataset);

        // 2. Get histograms
        // var histograms = await GetHistogramsAsync(options.CalibrationDataset);

        // 3. Select best ranges

        // 4. Assign ranges
        AssignRanges(ranges);
    }

    private async Task RunPassAsync(ICalibrationDatasetProvider calibrationDataset, Action<IReadOnlyDictionary<ENode, Tensor>> func)
    {
        await foreach (var sample in calibrationDataset.Samples)
        {
            var evaluator = new CalibrationEvaluator(sample, _rangeOfs);
            var values = evaluator.Evaluate();
            func(values);
        }
    }

    private async Task<IDictionary<ENode, ValueRange<float>>> GetRangesAsync(ICalibrationDatasetProvider calibrationDataset)
    {
        var ranges = new Dictionary<ENode, ValueRange<float>>();
        await RunPassAsync(calibrationDataset, values =>
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

    private Task<QuantizeHistogram> GetHistogramsAsync(ICalibrationDatasetProvider calibrationDataset)
    {
        throw new NotImplementedException();
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

        _graph.Rebuild();
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
                _rangeOfs.Add((ENode)match.Root);
            }
        }
    }
}
