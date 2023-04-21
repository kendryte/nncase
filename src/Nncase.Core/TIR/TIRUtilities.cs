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
    /// Compute the sub no pad bounds.
    /// 
    /// </summary>
    public static IReadOnlyList<TIR.Range> ComputeSubNoPadBounds(IReadOnlyList<TIR.Range> bounds, IReadOnlyList<TIR.Range> sub_bounds, IReadOnlyList<(IR.Expr Before, IR.Expr After)> paddings, IReadOnlyList<(IR.Expr Before, IR.Expr After)> sub_paddings, int? promote)
    {
        var subNoPadbounds = new TIR.Range[bounds.Count];
        for (int i = 0; i < bounds.Count; i++)
        {
            var bound = bounds[i];
            var sub_bound = sub_bounds[i];
            var (before, after) = paddings[i];
            var (before1, after1) = sub_paddings[i];

            /*  
              note 这里暂时用一种错误方式正确运行, 实际需要知道每个buffer在全局promote的维度和他的bounds是不是有关系. 
              因为通常提升都是-1,或者3. 对于act这种两维的参数正好忽略.
            */
            if (promote is int promoteInt && (promoteInt == -1 || i >= promoteInt))
            {
                var new_start = sub_bound.Start + before1;
                var new_stop = sub_bound.Stop - after1;
                subNoPadbounds[i] = new(new_start, new_stop, bound.Step);
            }
            else
            {
                var new_start = sub_bound.Start + before1 - (bound.Start + before); // 这里应该加一个偏移
                var new_stop = new_start + sub_bound.Stop - (sub_bound.Start + before1) - after1;
                subNoPadbounds[i] = new(new_start, new_stop, bound.Step);
            }
        }
        return subNoPadbounds;
    }

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
