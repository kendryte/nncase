// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.Passes;
using Nncase.PatternMatch;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Quantization;

internal partial class QuantizerAdaRound
{
    private readonly IEGraph _graph;
    private readonly List<ENode> _rangeOfs = new List<ENode>();
    private readonly List<ENode> _childrenOfRangeOfs = new List<ENode>();
    private readonly CompileSession _compileSession;

    public QuantizerAdaRound(IEGraph graph, CompileSession compileSession)
    {
        _graph = graph;
        _compileSession = compileSession;
        MarkRangeOfs();
    }

    public async Task RunAsync()
    {
        var quantOptions = _compileSession.CompileOptions.QuantizeOptions;
        if (quantOptions.UseAdaRound && quantOptions.CalibrationDataset == null)
        {
            throw new ArgumentNullException(nameof(quantOptions.CalibrationDataset));
        }

        if (quantOptions.UseAdaRound)
        {
            await _compileSession.Target.AdaRoundWeights(quantOptions.CalibrationDataset!, _rangeOfs, _childrenOfRangeOfs, quantOptions);
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
                var rangeOfMarker = (ENode)match.Root;

                // if (!_rangeOfs.Contains(_rangeOfMarker.Children[1].Nodes[0]))
                _rangeOfs.Add(rangeOfMarker.Children[1].Nodes[0]);

                // if (!_childrenOfRangeOfs.Contains(_rangeOfMarker.Children[0].Nodes[0]))
                _childrenOfRangeOfs.Add(rangeOfMarker.Children[0].Nodes[0]);
            }
        }
    }
}
