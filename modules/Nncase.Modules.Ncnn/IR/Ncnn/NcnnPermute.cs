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
/// Permute expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class NcnnPermute : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(NcnnPermute), 0, "input");

    /// <summary>
    /// Gets OrderType of NcnnPermute.
    /// </summary>
    public int OrderType { get; }

    /// <summary>
    /// Gets perm of source transpose used to eval.
    /// </summary>
    public int[] Perm { get; }

    /// <inheritdoc/>
    public override string DisplayProperty()
    {
        return $"OrderType:{OrderType}";
    }
}
