// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;

namespace Nncase.Schedule;

public interface IMicroKernelInfoProvider
{
    MicroKernelInfo GetInfo(Op op, IR.Affine.AffineDim[] domain, IR.Affine.AffineMap[] accessMaps, int[][] bufferShapes, ITargetOptions targetOptions);
}

/// <summary>
/// micro kernel infomation for auto tiling.
/// </summary>
/// <param name="Primitives"> primitives[i] is tileVar[i]'s minimal factor.</param>
/// <param name="Multiplier"> multiplier[i] is tileVar[i]'s search range.</param>
/// <param name="ReadBandWidth"> read from l1 bandwidth.</param>
/// <param name="WriteBandWidth"> write to l1 bandwidth.</param>
public record MicroKernelInfo(int[] Primitives, ValueRange<int>[] Multiplier, int ReadBandWidth, int WriteBandWidth)
{
}
