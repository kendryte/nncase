// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.TIR;

/// <summary>
/// tir helper.
/// </summary>
public static class TIRUtilities
{
    /// <summary>
    /// give the bounds and shape, compute paddings.
    /// </summary>
    public static IReadOnlyList<(IR.Expr Before, IR.Expr After)> ComputePaddings(IReadOnlyList<TIR.Range> bounds, IR.Shape shape) =>
    bounds.Select((bound, i) =>
      ((IR.Expr)IR.F.Math.Max(-bound.Start, 0), (IR.Expr)IR.F.Math.Max(bound.Stop - shape[i].FixedValue, 0))).ToArray();

    /// <summary>
    /// give two bounds compute paddings.
    /// </summary>
    public static IReadOnlyList<(IR.Expr Before, IR.Expr After)> ComputePaddings(IReadOnlyList<TIR.Range> bounds, IReadOnlyList<TIR.Range> target_bounds)
      => bounds.Zip(target_bounds).
        Select(it =>
          ((IR.Expr)IR.F.Math.Max(-it.First.Start, 0),
           (IR.Expr)IR.F.Math.Max(it.First.Stop - (it.Second.Stop - it.Second.Start), 0))).ToArray();

    /// <summary>
    /// compute the no padding bounds.
    /// </summary>
    public static IReadOnlyList<TIR.Range> ComputeNoPadBounds(IReadOnlyList<TIR.Range> bounds, IReadOnlyList<(IR.Expr Before, IR.Expr After)> paddings) =>
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
    public static IReadOnlyList<TIR.Range> ComputeBounds(IReadOnlyList<TIR.Range> sub_no_pad_bounds, IReadOnlyList<TIR.Range> bounds, IReadOnlyList<(IR.Expr Before, IR.Expr After)> paddings)
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
    public static IReadOnlyList<TIR.Range> ClampBounds(IReadOnlyList<TIR.Range> bounds, IR.Shape shape) =>
      bounds.Zip(shape).Select(
        t => new TIR.Range(
            IR.F.Math.Max(0, t.First.Start),
            IR.F.Math.Min(t.Second.FixedValue, t.First.Stop),
            t.First.Step)).ToArray();
}
