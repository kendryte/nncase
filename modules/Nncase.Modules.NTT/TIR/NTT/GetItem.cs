// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.TIR.NTT;

/// <summary>
/// Gather expression.
/// </summary>
public sealed partial class GetItem : NTTKernelOp
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(GetItem), 0, "input", ParameterKind.Input);

    /// <summary>
    /// Gets index.
    /// </summary>
    public static readonly ParameterInfo Index = new(typeof(GetItem), 1, "index", IsDimensionType() | IsShapeType(), ParameterKind.Input);

    /// <summary>
    /// Gets index.
    /// </summary>
    public static readonly ParameterInfo Output = new(typeof(GetItem), 2, "output");
}
