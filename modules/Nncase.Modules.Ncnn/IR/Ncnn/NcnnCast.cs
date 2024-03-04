// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.ArgsStruct;
using Nncase.PatternMatch;

namespace Nncase.IR.Ncnn;

/// <summary>
/// Cast expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class NcnnCast : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(NcnnCast), 0, "input");

    /// <summary>
    /// Gets FromType of Cast.
    /// </summary>
    public int FromType { get; }

    /// <summary>
    /// Gets ToType of Cast.
    /// </summary>
    public int ToType { get; }

    /// <inheritdoc/>
    public override string DisplayProperty()
    {
        return $"FromType: {FromType}, ToType: {ToType}";

    }
}
