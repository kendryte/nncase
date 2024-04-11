// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.TIR.CPU;

/// <summary>
/// ResizeImage expression.
/// </summary>
public sealed partial class ResizeImage : CPUKernelOp
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(ResizeImage), 0, "input", HasRank(r => r >= 2, "RanK >= 2"), ParameterKind.Input);

    public static readonly ParameterInfo Output = new(typeof(ResizeImage), 1, "output", HasRank(r => r >= 2, "RanK >= 2"), ParameterKind.Input);

    public IRArray<int> PackedAxes { get; }

    public IRArray<int> PadedNums { get; }

    public IRArray<int> NewSize { get; }

    public ImageResizeMode ResizeMode { get; }

    public ImageResizeTransformationMode TransformationMode { get; }

    public ImageResizeNearestMode NearestMode { get; }
}
