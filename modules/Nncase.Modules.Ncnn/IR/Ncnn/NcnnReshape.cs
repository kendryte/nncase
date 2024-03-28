// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.PatternMatch;

namespace Nncase.IR.Ncnn;

/// <summary>
/// Reshape expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class NcnnReshape : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(NcnnReshape), 0, "input");

    /// <summary>
    /// Gets Shape of Ncnn Reshape.
    /// </summary>
    public int[] Shape { get; }

    /// <inheritdoc/>
    public override string DisplayProperty()
    {
        return $"shape:{string.Join(",", Shape)}";
    }
}
