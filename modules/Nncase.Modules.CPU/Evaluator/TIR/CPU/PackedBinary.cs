// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.Schedule;
using Nncase.TIR.CPU;
using Nncase.Utilities;
using OrtKISharp;

namespace Nncase.Evaluator.TIR.CPU;

public sealed class PackedBinaryEvaluator : ITypeInferencer<PackedBinary>, IKernelInfoEvaluator<PackedBinary>
{
    public IRType Visit(ITypeInferenceContext context, PackedBinary target) => TupleType.Void;

    public MicroKernelInfo Visit(PackedBinary op, AffineDim[] domain, AffineMap[] accessMaps, int[][] bufferShapes, ITargetOptions targetOptions)
    {
        return new(Enumerable.Repeat(1, domain.Length).ToArray(), Enumerable.Repeat(new ValueRange<int>(1, int.MaxValue), domain.Length).ToArray(), 128, 128);
    }
}
