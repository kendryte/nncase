// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.IR.NN;

/// <summary>
/// GridSample expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class GridSample : Op
{
    public static readonly ParameterInfo Input = new(typeof(GridSample), 0, "input", HasRank(r => r >= 4, "RanK >= 4"), ParameterKind.Input);

    public static readonly ParameterInfo Grid = new(typeof(GridSample), 1, "grid", HasRank(r => r >= 4, "RanK >= 4"), ParameterKind.Input);

    public GridSampleAlignCorners AlignCorners { get; }

    public GridSampleMode Mode { get; }

    public GridSamplePaddingMode PaddingMode { get; }

    /// <inheritdoc/>
    public override string DisplayProperty() => $"GridSampleAlignCorners.{AlignCorners}, GridSampleMode.{Mode}, GridSamplePaddingMode.{PaddingMode}";
}
