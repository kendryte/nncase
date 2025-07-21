// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.IR.Tensors;

/// <summary>
/// Broadcast expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class Bitcast : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(Bitcast), 0, "input", ParameterKind.Input);

    public DataType NewType { get; }

    /// <inheritdoc/>
    public override string DisplayProperty() => $"{NewType.GetCSharpName()}";
}
