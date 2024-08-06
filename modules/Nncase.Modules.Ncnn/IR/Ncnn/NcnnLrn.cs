﻿// Copyright (c) Canaan Inc. All rights reserved.
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
/// Gets expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class NcnnLRN : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(NcnnLRN), 0, "input");

    /// <summary>
    /// Gets alpha of Ncnn LRN.
    /// </summary>
    public float Alpha { get; }

    /// <summary>
    /// Gets Beta of Ncnn LRN.
    /// </summary>
    public float Beta { get; }

    /// <summary>
    /// Gets Bias of Ncnn LRN.
    /// </summary>
    public float Bias { get; }

    /// <summary>
    /// Gets Size of Ncnn LRN.
    /// </summary>
    public int Size { get; }

    /// <inheritdoc/>
    public override string DisplayProperty()
    {
        return $"alpha={Alpha}, beta={Beta}, bias={Bias}, size={Size}";
    }
}
