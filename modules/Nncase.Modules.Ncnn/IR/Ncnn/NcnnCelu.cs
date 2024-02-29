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
/// Celu expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class NcnnCelu : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(NcnnCelu), 0, "input");

    /// <summary>
    /// Gets Alpha of Ncnn Celu.
    /// </summary>
    public float Alpha { get; }

    /// <inheritdoc/>
    public override string DisplayProperty()
    {
        return $"{Alpha}";
    }
}
