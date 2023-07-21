// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.IR.Buffers;

/// <summary>
/// Gets input.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class Uninitialized : Op
{
    /// <summary>
    /// the shape.
    /// </summary>
    public static readonly ParameterInfo Shape = new(typeof(Uninitialized), 0, "shape", IsIntegral() & IsTensor() & HasRank(1));

    public DataType DType { get; }

    public TIR.MemoryLocation MemoryLocation { get; }

    /// <inheritdoc/>
    public override bool CanFoldConstCall => false;

    /// <inheritdoc/>
    public override string DisplayProperty() => $"{DType.GetCSharpName()}, MemoryLocation.{MemoryLocation}";
}
