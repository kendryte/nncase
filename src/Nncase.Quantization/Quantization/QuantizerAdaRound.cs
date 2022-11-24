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

internal partial class QuantizerAdaRound
{
    private readonly EGraph _graph;
    private readonly List<ENode> _rangeOfs = new List<ENode>();
    private readonly List<ENode> _childrenOfRangeOfs = new List<ENode>();
    private readonly RunPassOptions _passOptions;

    public QuantizerAdaRound(EGraph graph, RunPassOptions passOptions)
    {
        _graph = graph;
        _passOptions = passOptions;
        MarkRangeOfs();
    }

    public async Task RunAsync(RunPassOptions options)
    {
        var quantOptions = options.CompileOptions.QuantizeOptions!;
        if (quantOptions.CalibrationDataset == null)
        {
            throw new ArgumentNullException(nameof(quantOptions.CalibrationDataset));
        }

        if (quantOptions.UseAdaRound)
        {
            await options.Target.AdaRoundWeights(quantOptions.CalibrationDataset, options.Target, _rangeOfs, _childrenOfRangeOfs, _passOptions);
        }

        _graph.Rebuild();
    }

    /// <summary>
    /// collec all rangeof enode.
    /// </summary>
    private void MarkRangeOfs()
    {
        if (EGraphMatcher.TryMatchRoot(_graph.Nodes, IsRangeOfMarker(IsWildcard(), IsConst()), out var matches))
        {
            // there are no rangeOfs and childrenOfRangeOfs actually, rangeOfs has been folded into const, use these names here is because of old habit.
            foreach (var match in matches)
            {
                var _rangeOfMarker = (ENode)match.Root;
                //if (!_rangeOfs.Contains(_rangeOfMarker.Children[1].Nodes[0]))
                _rangeOfs.Add(_rangeOfMarker.Children[1].Nodes[0]);
                //if (!_childrenOfRangeOfs.Contains(_rangeOfMarker.Children[0].Nodes[0]))
                _childrenOfRangeOfs.Add(_rangeOfMarker.Children[0].Nodes[0]);
            }
        }
    }
}