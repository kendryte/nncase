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
/// Dequantize expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class NcnnDequantize : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(NcnnDequantize), 0, "input");

    /// <summary>
    /// Gets scale of Ncnn Dequantize.
    /// </summary>
    public float[] Scale { get; }

    /// <summary>
    /// Gets scale of Ncnn Dequantize.
    /// </summary>
    public float[] Bias { get; }

    /// <inheritdoc/>
    public override string DisplayProperty()
    {
        return $"scale_size: {Scale.Length}, bias_size: {Bias.Length}";
    }
}
