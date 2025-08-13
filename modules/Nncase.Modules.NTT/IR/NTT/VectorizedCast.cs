// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.PatternMatch;

namespace Nncase.IR.NTT;

/// <summary>
/// Cast expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class VectorizedCast : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(VectorizedCast), 0, "input", ParameterKind.Input);

    public static readonly ParameterInfo PostOps = new(typeof(VectorizedCast), 1, "post_ops", ParameterKind.Attribute);

    public DataType NewType { get; }

    public CastMode CastMode { get; }

    public IRArray<int> VectorizeAxes { get; }

    /// <inheritdoc/>
    public override string DisplayProperty() => $"{NewType.GetCSharpName()}, CastMode.{CastMode}, VectorizeAxes: {{{string.Join(",", VectorizeAxes.IsDefaultOrEmpty ? Array.Empty<int>() : VectorizeAxes.ToArray())}}}";
}
