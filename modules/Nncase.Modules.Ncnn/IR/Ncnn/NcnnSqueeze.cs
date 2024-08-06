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
/// Squeeze expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class NcnnSqueeze : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(NcnnSqueeze), 0, "input");

    /// <summary>
    /// Gets dims of Ncnn Squeeze.
    /// </summary>
    public int[] Dims { get; }

    /// <inheritdoc/>
    public override string DisplayProperty()
    {
        return $"dims:{string.Join(",", Dims)}";
    }
}
