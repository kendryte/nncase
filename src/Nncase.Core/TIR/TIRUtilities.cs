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
/// tir helper
/// </summary>
public static class TIRUtilities
{
    /// <summary>
    /// give the bounds and shape, compute paddings.
    /// </summary>
    /// <param name="bounds"></param>
    /// <param name="shape"></param>
    /// <returns></returns>
    public static IEnumerable<(IR.Expr Before, IR.Expr After)> ComputePaddings(IEnumerable<TIR.Range> bounds, IR.Shape shape) =>
    bounds.Select((bound, i) =>
      ((IR.Expr)IR.F.Math.Max(-bound.Start, 0), (IR.Expr)IR.F.Math.Max(bound.Stop - shape[i].FixedValue, 0)));

    /// <summary>
    /// give two bounds compute paddings.
    /// </summary>
    /// <param name="bounds"></param>
    /// <param name="target_bounds"></param>
    /// <returns></returns>
    public static IEnumerable<(IR.Expr Before, IR.Expr After)> ComputePaddings(IEnumerable<TIR.Range> bounds, IEnumerable<TIR.Range> target_bounds)
      => bounds.Zip(target_bounds).
        Select(it =>
          ((IR.Expr)IR.F.Math.Max(-it.Item1.Start, 0),
           (IR.Expr)IR.F.Math.Max(it.Item1.Stop - (it.Item2.Stop - it.Item2.Start), 0)));

    /// <summary>
    /// compute the no padding bounds
    /// </summary>
    /// <param name="bounds"></param>
    /// <param name="paddings"></param>
    /// <returns></returns>
    public static IEnumerable<TIR.Range> ComputeNoPadBounds(IEnumerable<TIR.Range> bounds, IEnumerable<(IR.Expr Before, IR.Expr After)> paddings) =>
      bounds.Zip(paddings).Select(t =>
      {
          var bound = t.Item1;
          var pad = t.Item2;
          // var start = bound.Start - pad.Before;
          // var end = bound.Stop - pad.After;
          // glb 的start和end分别算
          return new TIR.Range(bound.Start - bound.Start, bound.Stop - bound.Start - (pad.Before + pad.After), bound.Step);
      });

    /// <summary>
    /// Compute the sub no pad bounds 
    /// </summary>
    public static IEnumerable<TIR.Range> ComputeSubNoPadBounds(IEnumerable<TIR.Range> bounds, IEnumerable<TIR.Range> sub_bounds, IEnumerable<(IR.Expr Before, IR.Expr After)> paddings,
    IEnumerable<(IR.Expr Before, IR.Expr After)> sub_paddings) =>
      bounds.Zip(sub_bounds).Zip(paddings, sub_paddings).Select(t =>
      {
          var (bound, sub_bounds) = t.First;
          var pad = t.Second;
          var sub_pad = t.Third;
          var new_start = (sub_bounds.Start + sub_pad.Before) - (bound.Start + pad.Before);
          var new_stop = new_start + sub_bounds.Stop - (sub_bounds.Start + sub_pad.Item1) - sub_pad.Item2;
          return (TIR.Range)(new_start, new_stop, bound.Step);
      });

    /// <summary>
    /// give the sub no pad bounds, then get the current bounds.
    /// </summary>
    /// <param name="sub_no_pad_bounds"></param>
    /// <param name="bounds"></param>
    /// <param name="paddings"></param>
    /// <returns></returns>
    public static IEnumerable<TIR.Range> ComputeBounds(IEnumerable<TIR.Range> sub_no_pad_bounds, IEnumerable<TIR.Range> bounds, IEnumerable<(IR.Expr Before, IR.Expr After)> paddings)
    {
        return sub_no_pad_bounds.Zip(bounds, paddings).Select(t =>
        {
            var rg = t.First;
            var bound = t.Second;
            var padding = t.Third;
            var start = bound.Start + padding.Before + rg.Start;
            var stop = (rg.Stop - rg.Start) + start;
            return new TIR.Range(start, stop, rg.Step);
        });
    }

    /// <summary>
    /// clamp bounds by given shape.
    /// </summary>
    /// <param name="bounds"></param>
    /// <param name="shape"></param>
    /// <returns></returns>
    public static IEnumerable<TIR.Range> ClampBounds(IEnumerable<TIR.Range> bounds, IR.Shape shape) =>
      bounds.Zip(shape).Select(
        t => new TIR.Range(IR.F.Math.Max(0, t.Item1.Start),
                           IR.F.Math.Min(t.Item2.FixedValue, t.Item1.Stop),
                           t.Item1.Step));
}