﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Immutable;
using Nncase.IR;

namespace Nncase.Schedule;

public interface IMicroKernelInfoProvider
{
    MicroKernelInfo GetInfo(Op op, MicroKernelContext context);
}

/// <summary>
/// each buffer info.
/// </summary>
/// <param name="ReadBandWidth">read bandwidth from l1.</param>
/// <param name="WriteBandWidth">write bandwidth from l1.</param>
/// <param name="State">state.</param>
public record MicroKernelBufferInfo(int ReadBandWidth, int WriteBandWidth, MicroKernelBufferInfo.BufferState State)
{
    [Flags]
    public enum BufferState : byte
    {
        Read = 1 << 1,
        Write = 1 << 2,
    }
}

/// <summary>
/// micro kernel infomation for auto tiling.
/// </summary>
public record MicroKernelInfo(int[] Primitives, ValueRange<long>[] Multipliers, MicroKernelBufferInfo[] BufferInfos, Func<Google.OrTools.ConstraintSolver.IntExpr[][], Google.OrTools.ConstraintSolver.Solver, MicroKernelContext, Google.OrTools.ConstraintSolver.IntExpr> GetComputeCycle)
{
}

public record MicroKernelContext(Op Op, ImmutableArray<IR.Affine.AffineMap> AccessMaps, ImmutableArray<ImmutableArray<long>> BufferShapes, ITargetOptions TargetOptions)
{
}
