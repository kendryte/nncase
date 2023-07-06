// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Immutable;
using System.Runtime.CompilerServices;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.TIR.CPU;

namespace Nncase.Passes.Tile;

/// <summary>
/// the multi Fusion checker.
/// </summary>
internal sealed class MultiFusionChecker : IFusionChecker
{
    private readonly List<(MultiLayerFusionConverter, int[], BufferSchedule.ScheduledResponse)> _caches = new();

    private readonly TileOptions _tileOptions;

    public MultiFusionChecker(TileOptions tileOptions)
    {
        _tileOptions = tileOptions;
    }

    [Flags]
    public enum DeviceKind
    {
        Load,
        Store,
        Mfu,
        Tcu,
        None,
    }

    public TIR.PrimFunction Convert(RunPassContext passOptions)
    {
        var (convertVisitor, final_tile_size, response) = _caches.First();
        if (DumpScope.Current.IsEnabled(DumpFlags.PassIR))
        {
            response.Dump($"{response.LogicalPrimfunc.Name}_{string.Join("_", final_tile_size)}", convertVisitor.GetType().Name);
        }

        return convertVisitor.BuildPhysicalPrimFunc(final_tile_size, response.SchedCandidate, response.LogicalPrimfunc);
    }

    public bool Check(Fusion fusion, RunPassContext passOptions)
    {
        // 1. check all conv2d weights size less than glb size
        var visitor = new MultiFusionPreCheckVisitor();
        visitor.Visit(fusion);
        if (visitor.AllWeightSizeInBytes > ExtCompilerServices.Env.GlbSize)
        {
            return false;
        }

        // note not support conv2d transpose in layer group.
        if (visitor.CountCallOp<IR.K510.GNNEConv2DTranspose>() > 0)
        {
            return false;
        }

        var curTileOptions = _tileOptions;
        if ((visitor.DeviceUsage[DeviceKind.Mfu], visitor.DeviceUsage[DeviceKind.Tcu]) switch
        {
            (> 1, > 1) => true,
            (> 1, 1) => true,
            (1, > 1) => true,
            _ => false,
        })
        {
            curTileOptions = curTileOptions with { PingPongNum = 3 };
        }

        // 2. try convert
        var convertVisitor = new MultiLayerFusionConverter(curTileOptions); // note the grouped fusion must pingpong input.
        var bodySeq = convertVisitor.Visit(fusion);

        // 3. search the tile size
        var originLogicalPrimFunc = convertVisitor.BuildLogicalPrimFunc(bodySeq);

        var output_shape = fusion.Body.CheckedShape.ToValueArray();
        var search_space = convertVisitor.BoundsInferGraph.RootTileStep.ToArray();
        var candidate_tile_size = convertVisitor.SearchTileSize(TileOhSearchGenerator(curTileOptions, search_space, convertVisitor.BoundsInferGraph.RootPerm.ToArray()), originLogicalPrimFunc, curTileOptions.MultiWorkers, false, out var sched_response);
        if (!candidate_tile_size.Any())
        {
            return false;
        }

        int[] final_tile_size = new int[candidate_tile_size.Length];
        if (convertVisitor.BalanceTileSize(candidate_tile_size, search_space))
        {
            final_tile_size = convertVisitor.SearchTileSize(new TargetTileGenerator(candidate_tile_size), originLogicalPrimFunc, curTileOptions.MultiWorkers, true, out sched_response);
        }
        else
        {
            Array.Copy(candidate_tile_size, final_tile_size, candidate_tile_size.Length);
        }

        // 5. check the input load usage and compute overlap
        var input_shape = fusion.Parameters[0].CheckedShape.ToValueArray();
        var each_axis_tile_nums = final_tile_size.Zip(output_shape).Select(p => (int)System.Math.Ceiling(p.Second / (float)p.First)).ToArray();
        var total_tile_nums = TensorUtilities.GetProduct(each_axis_tile_nums);
        if (total_tile_nums > 1)
        {
            var clamp = (TIR.K510.Segment seg, int i) =>
            {
                return new TIR.K510.Segment(Math.Max(0, seg.Start), Math.Min(input_shape[i], seg.Stop), 1);
            };

            var first_segment = convertVisitor.BoundsInferGraph[convertVisitor.VarToKeyMap[fusion.Parameters[0]]].
              Eval(final_tile_size.Select(t => new TIR.K510.Segment(0, t, 1)).ToArray()).
              Select((s, i) => clamp(s, i)).
              ToArray();

            int first_split_axis = 0;
            for (int i = input_shape.Length - 1; i >= 0; i--)
            {
                if (first_segment[i].Length != input_shape[i])
                {
                    first_split_axis = i;
                    break;
                }
            }

            // when once load less than load burst, false
            var burst_load_data = TensorUtilities.GetProduct(first_segment.Skip(first_split_axis).Select(s => s.Length).ToArray());
            if (burst_load_data < ExtCompilerServices.Env.LoadBurst)
            {
                return false;
            }

            var second_segment = convertVisitor.BoundsInferGraph[convertVisitor.VarToKeyMap[fusion.Parameters[0]]].Eval(final_tile_size.Select((t, i) =>
                  t < output_shape[i] ?
                  new TIR.K510.Segment(t, System.Math.Min(t * 2, output_shape[i]), 1) :
                  new TIR.K510.Segment(0, t, 1)).ToArray()).
                  Select((s, i) => clamp(s, i)).
                  ToArray();

            // Todo 因为我无法知道在当前维度切分会影响哪个维度的变化, 比如带有transpose的, 可能我在c上切,只影响 h w. 所以直接计算所有的的交集
            var overlaps = first_segment.Zip(second_segment).Select(p => p.First.Intersect(p.Second)).ToArray();
            if (Array.IndexOf(convertVisitor.BoundsInferGraph.RootPerm.ToArray(), TIR.K510.NamedAxis.H) is int h && h != -1)
            {
                // 如果只在h上切分, 只需要考虑h上的overlap有没有超过0.3
                if (overlaps[h] > (input_shape[h] * 0.3))
                {
                    return false;
                }
            }
            else
            {
                if (TensorUtilities.GetProduct(overlaps) > TensorUtilities.GetProduct(input_shape) * 0.3)
                {
                    return false;
                }
            }
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
    /// 只在oh上切分.
    /// </remark>
    /// </summary>
    private ISearchTileGenerator TileOhSearchGenerator(TileOptions tileOptions, Segment[] search_spaces, TIR.K510.NamedAxis[] rootPerm)
    {
        if (Array.IndexOf(rootPerm, TIR.K510.NamedAxis.C) is int c && c != -1)
        {
            search_spaces[c].Start = search_spaces[c].Stop; // not tile oc
        }

        if (Array.IndexOf(rootPerm, TIR.K510.NamedAxis.H) is int h && h != -1)
        {
            // 因为在h上切分, 如果ping pong那么需要限制大小
            if (tileOptions.PingPong)
            {
                if (search_spaces[h].ClampStop(2, out var new_h_seg))
                {
                    // assume tile h must > 8 for tcu use.
                    search_spaces[h] = new Segment(Math.Min(Math.Max(search_spaces[h].Step, ExtCompilerServices.Env.TcuActNum), new_h_seg.Stop), new_h_seg.Stop, new_h_seg.Step);
                }
            }
        }

        if (Array.IndexOf(rootPerm, TIR.K510.NamedAxis.W) is int w && w != -1)
        {
            search_spaces[w].Start = Math.Min(search_spaces[w].Stop, Math.Max(search_spaces[w].Step, ExtCompilerServices.Env.PuWidth)); // no tile w
        }

        // 如果有perm, 那就是 c w h n 方式搜, 没有perm就是从最后搜到最前 所以在有ping pong的时候需要限制切分.
        if (rootPerm.All(r => r == NamedAxis.UnKnow) && tileOptions.PingPong)
        {
            if (search_spaces[^1].ClampStop(2, out var new_seg))
            {
                search_spaces[^1] = new_seg;
            }
        }

        return new DefaultSearchTileGenerator(search_spaces, rootPerm);
    }

    internal sealed class MultiFusionPreCheckVisitor : ExprVisitor<bool, bool>
    {
        public Dictionary<DeviceKind, int> DeviceUsage { get; } = new()
        {
            { DeviceKind.Load, 0 },
            { DeviceKind.Store, 0 },
            { DeviceKind.Mfu, 0 },
            { DeviceKind.Tcu, 0 },
            { DeviceKind.None, 0 },
        };

        public int AllWeightSizeInBytes { get; private set; }

        public int CountCallOp<T>()
          where T : Op
        {
            return ExprMemo.Keys.Count(e => e is Call { Target: Op t } && t.GetType() == typeof(T));
        }

        protected override bool DefaultVisitLeaf(Expr expr) => true;

        protected override bool VisitLeafCall(Call expr)
        {
            if (expr is Call { Target: IR.K510.GNNEConv2D } && expr[IR.K510.GNNEConv2D.Weights] is Expr weights)
            {
                AllWeightSizeInBytes += weights.CheckedShape.Prod().FixedValue * weights.CheckedDataType.SizeInBytes;
            }

            DeviceUsage[GetDeviceType(expr.Target)]++;
            return true;
        }

        private static DeviceKind GetDeviceType(Expr op) => op switch
        {
            IR.K510.GNNELoad => DeviceKind.Load,
            IR.K510.GNNEStore => DeviceKind.Store,
            IR.K510.GNNEConv2D or IR.K510.GNNEActivation => DeviceKind.Tcu,
            IR.K510.GNNEReduce or IR.K510.GNNEMeshNet or IR.K510.GNNETranspose or IR.K510.GNNEPdpReduce or IR.K510.GNNECrop => DeviceKind.Mfu,
            _ => DeviceKind.None,
        };
    }
}
