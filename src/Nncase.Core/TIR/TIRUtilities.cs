// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.Shapes;

namespace Nncase.TIR;

/// <summary>
/// tir helper.
/// </summary>
public static class TIRUtilities
{
    /// <summary>
    /// give the bounds and shape, compute paddings.
    /// </summary>
    public static Paddings ComputePaddings(IReadOnlyList<TIR.Range> bounds, RankedShape shape) => new Paddings(
      bounds.Zip(shape).Select((t, i) =>
        new Padding(Dimension.Max(-t.First.Start, 0), Dimension.Max(t.First.Stop - t.Second, 0))).ToArray());

    /// <summary>
    /// give two bounds compute paddings.
    /// </summary>
    public static Paddings ComputePaddings(IReadOnlyList<TIR.Range> bounds, IReadOnlyList<TIR.Range> targetBounds)
      => new Paddings(bounds.Zip(targetBounds).
        Select(it =>
          new Padding(Dimension.Max(-it.First.Start, 0), Dimension.Max(it.First.Stop - (it.Second.Stop - it.Second.Start), 0))).ToArray());

    /// <summary>
    /// compute the no padding bounds.
    /// </summary>
    public static IReadOnlyList<TIR.Range> ComputeNoPadBounds(IReadOnlyList<TIR.Range> bounds, Paddings paddings) =>
      bounds.Zip(paddings).Select(t =>
      {
          var bound = t.First;
          var (before, after) = t.Second;

          // var start = bound.Start - pad.Before;
          // var end = bound.Stop - pad.After;
          // glb 的start和end分别算
          return new TIR.Range(bound.Start - bound.Start, bound.Stop - bound.Start - (before + after), bound.Step);
      }).ToArray();

    /// <summary>
    /// give the sub no pad bounds, then get the current bounds.
    /// </summary>
    public static IReadOnlyList<TIR.Range> ComputeBounds(IReadOnlyList<TIR.Range> sub_no_pad_bounds, IReadOnlyList<TIR.Range> bounds, Paddings paddings)
    {
        return sub_no_pad_bounds.Zip(bounds, paddings).Select(t =>
        {
            var rg = t.First;
            var bound = t.Second;
            var (before, after) = t.Third;
            var start = bound.Start + before + rg.Start;
            var stop = rg.Stop - rg.Start + start;
            return new TIR.Range(start, stop, rg.Step);
        }).ToArray();
    }

    /// <summary>
    /// clamp bounds by given shape.
    /// </summary>
    public static IReadOnlyList<TIR.Range> ClampBounds(IReadOnlyList<TIR.Range> bounds, RankedShape shape) =>
      bounds.Zip(shape).Select(
        t => new TIR.Range(
            Dimension.Max(0, t.First.Start),
            Dimension.Min(t.Second.FixedValue, t.First.Stop),
            t.First.Step)).ToArray();

    public static bool TryGetFixedRegions(TIR.BufferRegion region, out (long Start, long Stop, long Step)[] slice)
    {
        slice = new (long Start, long Stop, long Step)[region.Region.Length];
        for (int i = 0; i < region.Region.Length; i++)
        {
            var rg = region.Region[i];
            if (rg is not Range { Start: DimConst start, Stop: DimConst stop, Step: DimConst step })
            {
                return false;
            }

            slice[i] = (start.Value, stop.Value, step.Value);
        }

        return true;
    }
}
