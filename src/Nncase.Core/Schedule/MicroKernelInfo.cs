// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;

namespace Nncase.Schedule;

public interface IMicroKernelInfoProvider
{
    MicroKernelInfo GetInfo(Op op, IR.Affine.AffineDim[] domain, IR.Affine.AffineMap[] accessMaps, int[][] bufferShapes, ITargetOptions targetOptions);
}

public record MicroKernelInfo(int[] Primitives, ValueRange<int>[] Multiplier, int ReadBandWidth, int WriteBandWidth)
{
}
