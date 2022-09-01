// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.TIR;

/// <summary>
/// Representing the region of multi-dimensional buffer access.
/// NOTE the region can be negative, we can use negative calc the padding.
/// </summary>
/// <param name="Buffer">The buffer of the buffer region.</param>
/// <param name="Region">The region array of the buffer region.</param>
public sealed record BufferRegion(Buffer Buffer, IRArray<Range> Region) : Expr
{
    /// <summary>
    /// Create a BufferRegion which is full region of the given buffer.
    /// NOTE because of the each backend has different addr calc logic.
    /// </summary>
    /// <param name="buffer">The buffer to generate full BufferRegion.</param>
    /// <returns>The BufferRegion which covers all region of the given buffer.</returns>
    // public static BufferRegion All(PhysicalBuffer buffer) => new BufferRegion(buffer, new(buffer.Dimensions.Select(extent => new Range(0, extent, 1))));

    /// <summary>
    /// Get the RegionSize.
    /// </summary>
    // public Expr[] RegionSize => Region.Select(r => r.Stop - r.Start).ToArray();

    /// <summary>
    /// Get padding at the dim.
    /// </summary>
    /// <param name="dim"></param>
    /// <returns></returns>
    // public (Expr Before, Expr After) Padding(int dim) => (IR.F.Math.Max(-Region[dim].Start, 0), IR.F.Math.Max(Region[dim].Stop - Buffer.Dimensions[dim], 0));

    /// <summary>
    /// 获得新的buffer region.
    /// </summary>
    /// <param name="ranges"></param>
    /// <returns></returns>
    public BufferRegion this[params TIR.Range[] ranges]
    {
        get => new(Buffer, new(Region.Zip(ranges).Select(
            tp => tp.Second.Equals(System.Range.All) ?
                  tp.First :
                  tp.Second.Stop switch
                  {
                      // if stop is neg, add the shape
                      Call { Target: IR.Math.Unary { UnaryOp: UnaryOp.Neg } } => throw new NotSupportedException("Neg Region!"),
                      // else return the origin range.
                      _ => tp.Second
                  })));
    }
}
