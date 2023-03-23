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

namespace Nncase.IR.Tensors;

/// <summary>
/// Stack expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class Tile : Op
{
    /// <summary>
    /// Gets input\.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(Tile), 0, "input");

    /// <summary>
    /// Gets repeats.
    /// </summary>
    public static readonly ParameterInfo Repeats = new(typeof(Tile), 1, "repeats", HasRank(1) & HasDataType(DataTypes.Int64));
}
