// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.IR.Buffer;

/// <summary>
/// Gets input.
/// </summary>
[PatternFunctionalGenerator]
public sealed record Uninitialized(DataType DType, Schedule.MemoryLocation MemoryLocation) : Op
{
    /// <summary>
    /// the shape
    /// </summary>
    public static readonly ParameterInfo Shape = new(typeof(Uninitialized), 0, "shape", IsIntegral() & IsTensor() & HasRank(1));
}
