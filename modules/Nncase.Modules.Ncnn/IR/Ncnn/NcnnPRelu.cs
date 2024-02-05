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
/// PRelu expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class NcnnPRelu : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(NcnnPRelu), 0, "input");

    /// <summary>
    /// Gets Slope of Ncnn PRelu.
    /// </summary>
    public float[] Slope { get; }

    /// <inheritdoc/>
    public override string DisplayProperty()
    {
        return $"";
    }
}
