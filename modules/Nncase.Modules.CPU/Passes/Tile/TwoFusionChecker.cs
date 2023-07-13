// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

#if false
using System.Collections.Immutable;
using System.Runtime.CompilerServices;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.PatternMatch;
using Nncase.TIR.K510;
using static Nncase.PatternMatch.Utility;

[assembly: InternalsVisibleTo("Nncase.Tests.K510")]

namespace Nncase.Passes.Tile;

/// <summary>
/// the two Fusion checker.
/// 专门前一层可以在conv的ic上tiling的情况.
/// </summary>
internal sealed class TwoFusionChecker : IFusionChecker
{
    private readonly List<(LayerFusionConverter, int[], BufferSchedule.ScheduledResponse)> _caches = new();

    private readonly TileOptions _tileOptions;

    public TwoFusionChecker(TileOptions tileOptions)
    {
        _tileOptions = tileOptions;
    }

    /// <summary>
    /// Gets 匹配 conv2d + 非reduction.
    /// </summary>
    public static Pattern TwoFusionPattern { get; } = IsCallWildcard(
        null,
        IsOp<IR.K510.GNNEStore>(),
        IsCallWildcard(
          "conv2d",
          IsOp<IR.K510.GNNEConv2D>(),
          IsCallWildcard(null, IsOp<Op>("calleeOp", op => op is IR.K510.GNNEMeshNet or IR.K510.GNNEPdpReduce), IsCallWildcard(null, IsOp<IR.K510.GNNELoad>()))));

    public TIR.PrimFunction Convert(RunPassContext passOptions)
    {
        var (convertVisitor, final_tile_size, response) = _caches.First();
        if (DumpScope.Current.IsEnabled(DumpFlags.PassIR))
        {
            response.Dump($"{response.LogicalPrimfunc.Name}_{string.Join("_", final_tile_size)}", nameof(LayerFusionOcIcConverter));
        }

        return convertVisitor.BuildPhysicalPrimFunc(final_tile_size, response.SchedCandidate, response.LogicalPrimfunc);
    }

    public bool Check(Fusion fusion, RunPassContext passOptions)
    {
        // 1. try match pattern
        if (!CompilerServices.TryMatchRoot(fusion.Body, TwoFusionPattern, out var matchResult))
        {
            return false;
        }

        // 2. try convert
        var convertVisitor = new LayerFusionOcIcConverter(_tileOptions, TileUtilities.ChoiceTcuStrategy((Call)matchResult["conv2d"], out _), false); // note the grouped fusion must pingpong input.
        var bodySeq = convertVisitor.Visit(fusion);

        // 3. search the tile size
        var originLogicalPrimFunc = convertVisitor.BuildLogicalPrimFunc(bodySeq);
        _ = fusion.Body.CheckedShape.ToValueArray();
        var search_space = convertVisitor.OCBoundsInferGraph.RootTileStep.ToArray();
        search_space = search_space.Concat(new[] { new Segment(1, convertVisitor.Conv2DInShape[1], 1) }).ToArray();
        var candidate_tile_size = convertVisitor.SearchTileSize(
            SearchGenerator(search_space),
            originLogicalPrimFunc,
            _tileOptions.MultiWorkers,
            false,
            out var sched_response);
        if (!candidate_tile_size.Any())
        {
            return false;
        }

        if (_tileOptions.PingPong && convertVisitor.BalanceTileSize(candidate_tile_size, search_space))
        {
            _ = convertVisitor.SearchTileSize(new TargetTileGenerator(candidate_tile_size), originLogicalPrimFunc, _tileOptions.MultiWorkers, true, out sched_response);
        }
        else
        {
        }

        _caches.Add((convertVisitor, candidate_tile_size, sched_response));
        if (_caches.Count > 1)
        {
            _caches.RemoveAt(0);
        }

        return true;
    }

    /// <summary>
    /// <remark>
    /// do not tile on w dimension.
    /// </remark>
    /// </summary>
    /// <returns>.</returns>
    private ISearchTileGenerator SearchGenerator(Segment[] search_spaces)
    {
        var newSpaces = search_spaces.ToArray();

        // 这里就优先在ic上切分ping pong. 因为在ic上切分对于if来说都是不一样的.
        var ic = newSpaces.Length - 1;
        if (newSpaces[ic].ClampStop(2, out var new_seg))
        {
            newSpaces[ic] = new_seg;
        }

        // ic最小也得分两个tcu.
        newSpaces[ic].Start = Math.Min(ExtCompilerServices.Env.TcuActNum * ExtCompilerServices.Env.PuHeight, newSpaces[ic].Stop);

        var w = 3;
        newSpaces[w].Start = newSpaces[w].Stop; // no tile w

        return new QueuedSearchTileGenerator(newSpaces, g =>
        {
            g.Queue.Add((1, System.Math.Min(ExtCompilerServices.Env.PuWidth * ExtCompilerServices.Env.TcuActNum, g.UpperBounds[1]))); // oc
            g.Queue.Add((4, g.UpperBounds[4])); // ic
            g.Queue.Add((2, g.UpperBounds[2])); // h
            g.Queue.Add((1, g.UpperBounds[1])); // oc
        });
    }
}
#endif
