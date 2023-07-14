// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

#if false
using System.Runtime.CompilerServices;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.Passes.BufferSchedule;
using Nncase.TIR;
using Nncase.TIR.Builders;
using Nncase.TIR.CPU;
using Nncase.TIR.K510.Builders;
using MathF = Nncase.IR.F.Math;

namespace Nncase.Passes.Tile;

internal class MultiLayerFusionConverter : LayerFusionConverter
{
    public MultiLayerFusionConverter(TileOptions tileOptions)
    {
        TileOptions = tileOptions;
    }

    public override Fusion CurrentFusion { get; protected set; } = null!;

    public override IBoundsInferGraph BoundsInferGraph { get; protected set; } = null!;

    /// <summary>
    /// Gets or sets calc the loop count.
    /// </summary>
    public override Expr LoopCount { get; protected set; } = null!;

    public override Expr LoopCountOuter { get; protected set; } = null!; // LoopCount / TileOptions.PingPongNum;

    public override Expr LoopCountInner { get; protected set; } = null!; // LoopCount % TileOptions.PingPongNum;

    public override Expr Visit(Fusion fusion)
    {
        if (CurrentFusion is null)
        {
            CurrentFusion = fusion;
        }
        else
        {
            throw new InvalidOperationException("Can't Visit More Than One Fusion!");
        }

        // 0. init the fields
        var output_shape = Tile.TileUtilities.GetFusionRealOutputShape(fusion.Body);
        BoundsInferGraph = ExtCompilerServices.MakeBoundsInferGraph((Call)fusion.Body);
        TileSizeVars.AddRange(output_shape.Select((_, i) => new Var($"dim{i}_tile", new TensorType(DataTypes.Int32, Shape.Scalar))));

        // 1. make the tile gird loop
        NestedBlocks.AddRange(new[] { EAction.TileBlock("MainBlock") }.Concat(Enumerable.Range(0, TileSizeVars.Count).Select(i => EAction.TileBlock($"TileBlock_{i}"))));

        LoopDomains.AddRange(output_shape.Zip(TileSizeVars).Select(t => new TIR.Range(0, t.First, t.Second)));
        for (int i = 0; i < TileSizeVars.Count; i++)
        {
            NestedLoops.Add(T.ForLoop(out var loopVar, LoopDomains[i], LoopMode.Unrolled, $"loop_var_{i}"));
            LoopVars.Add(loopVar);
        }

        object lastBody = NestedBlocks[^1];
        for (int i = NestedLoops.Count - 1; i >= 0; i--)
        {
            lastBody = NestedLoops[i].Body(lastBody);
            lastBody = NestedBlocks[i].Body(lastBody);
        }

        // 2. create the bounds infer function input arguments with the new loop var.
        BoundsInferGraph.RootBounds = output_shape.Select((s, i) =>
         {
             var loopVar = LoopVars[i];
             return new TIR.Range(loopVar, IR.F.Math.Min(loopVar + TileSizeVars[i], s), 1);
         }).ToList();

        // 3. set up loop count
        var shape = new Expr[LoopVars.Count];
        var upbounds = CurrentFusion.Body.CheckedShape.ToValueArray();
        for (int j = LoopVars.Count - 1; j >= 0; j--)
        {
            shape[j] = TileUtilities.SplitTimes(upbounds[j], TileSizeVars[j]);
        }

        var strides = TensorUtilities.GetStrides(shape).ToArray();
        var indices = LoopVars.Select((v, j) => (Expr)(v / TileSizeVars[j])).ToArray();
        LoopCount = TensorUtilities.GetIndex(strides, indices);
        LoopCountOuter = LoopCount / TileOptions.PingPongNum;
        LoopCountInner = LoopCount % TileOptions.PingPongNum;

        return Visit((Call)fusion.Body, "root");
    }

    /// <summary>
    /// convert to the final prim func.
    /// </summary>
    /// <returns>.</returns>
    public PrimFunction VisitToPrimFunc(Fusion fusion)
    {
        // 1. visit the fusion
        var bodySeq = Visit(fusion);

        // 2. build the prim func with tile size vars.
        var logicalPrimFunc = BuildLogicalPrimFunc(bodySeq);

        // 3. seach the tiling size
        var search_spaces = BoundsInferGraph.RootTileStep.ToArray();
        ISearchTileGenerator tileGenerator;
        if (TileOptions.TargetTileSize.Any())
        {
            for (int i = 0; i < TileOptions.TargetTileSize.Length; i++)
            {
                System.Diagnostics.Trace.Assert(TileOptions.TargetTileSize[i] <= search_spaces[i].Stop);
            }

            tileGenerator = new TargetTileGenerator(TileOptions.TargetTileSize);
        }
        else
        {
            var perm = BoundsInferGraph.RootPerm.ToArray();

            // when ping pong all, clamp the upper bounds by perm order.
            if (TileOptions.PingPong)
            {
                var re_perm = perm.Zip(Enumerable.Range(0, perm.Length)).OrderBy(t => t.First).Select(t => t.Second).ToArray();

                var pp_axis = NamedAxis.H;

                // 如果已知维度, 那么在pp axis上进行切分
                if (Array.IndexOf(perm, NamedAxis.H) is var h && h != -1 && Array.IndexOf(perm, NamedAxis.W) is var w && w != -1)
                {
                    // if split the h will less than one burst, split on c.
                    if ((int)System.Math.Ceiling(search_spaces[h].Stop / (float)TileOptions.PingPongNum) * search_spaces[w].Stop < 128)
                    {
                        pp_axis = NamedAxis.C;
                    }
                    else
                    {
                        pp_axis = NamedAxis.H;
                    }
                }

                for (int i = 0; i < perm.Length; i++)
                {
                    var p = re_perm[i];
                    if (perm[i] == pp_axis && search_spaces[p].ClampStop(TileOptions.PingPongNum, out var new_seg))
                    {
                        search_spaces[p] = new_seg;
                        break;
                    }
                }
            }

            {
                if (Array.IndexOf(perm, NamedAxis.H) is var h && h != -1 && Array.IndexOf(perm, NamedAxis.W) is var w && w != -1)
                {
                    // if one layer output less than 128, don't split the hw
                    if (search_spaces[h].Stop * search_spaces[w].Stop < ExtCompilerServices.Env.LoadBurst)
                    {
                        search_spaces[h].Start = search_spaces[h].Stop;
                        search_spaces[w].Start = search_spaces[w].Stop;
                    }
                    else
                    {
                        search_spaces[h].Start = Math.Min(Math.Max(search_spaces[h].Step, ExtCompilerServices.Env.TcuActNum), search_spaces[h].Stop);
                        search_spaces[w].Start = Math.Min(Math.Max(search_spaces[w].Step, ExtCompilerServices.Env.PuWidth), search_spaces[w].Stop);
                    }
                }
            }

            tileGenerator = new DefaultSearchTileGenerator(search_spaces, BoundsInferGraph.RootPerm);
        }

        int[] candidate_tile_size = SearchTileSize(tileGenerator, logicalPrimFunc, TileOptions.MultiWorkers, false, out var response);
        if (!candidate_tile_size.Any())
        {
            throw new TileFailedException();
        }

        int[] final_tile_size = Array.Empty<int>();
        if (!TileOptions.TargetTileSize.Any() && TileOptions.PingPong && BalanceTileSize(candidate_tile_size, search_spaces))
        {
            final_tile_size = SearchTileSize(new TargetTileGenerator(candidate_tile_size), logicalPrimFunc, TileOptions.MultiWorkers, true, out response);
        }
        else
        {
            final_tile_size = candidate_tile_size;
        }

        if (DumpScope.Current.IsEnabled(DumpFlags.PassIR))
        {
            response.Dump($"{CurrentFusion.Name}_{string.Join("_", final_tile_size)}", GetType().Name);
        }

        // 4. the local logical buffer to phsy buffer
        return BuildPhysicalPrimFunc(final_tile_size, response.SchedCandidate, response.LogicalPrimfunc);
    }

    /// <summary>
    /// 1. if inner loop var > half, balance it.
    /// 2. find the highest axis loop var == up_bounds, split it.
    /// </summary>
    public override bool BalanceTileSize(int[] tile_size, Segment[] search_spaces)
    {
        bool changed = false;

        // balance tile
        for (int i = search_spaces.Length - 1; i >= 0; i--)
        {
            if (search_spaces[i].BalanceTile(tile_size[i], out var newTile))
            {
                tile_size[i] = newTile;
                return true;
            }
        }

        // force ping pong
        for (int i = 0; i < search_spaces.Length; i++)
        {
            if (search_spaces[i].ClampStop(2, out var new_seg) && new_seg.Stop < tile_size[i])
            {
                tile_size[i] = new_seg.Stop;
                return true;
            }
        }

        return changed;
    }
}
#endif
