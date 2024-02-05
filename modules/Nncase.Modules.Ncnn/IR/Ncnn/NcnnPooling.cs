// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.PatternMatch;
using Nncase.ArgsStruct;

namespace Nncase.IR.Ncnn;

/// <summary>
/// Pooling expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class NcnnPooling : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(NcnnPooling), 0, "input");

    /// <summary>
    /// Gets PoolingArgs of Ncnn Pooling.
    /// </summary>
    public PoolingArgs Args { get; }

    /// <inheritdoc/>
    public override string DisplayProperty()
    {
        return $"{Args.PoolingType}, Kernel: {Args.KernelW}-{Args.KernelH}, Stride: {Args.StrideW}-{Args.StrideH}, Padding: {Args.PadLeft}-{Args.PadRight}-{Args.PadTop}-{Args.PadBottom}, CeilMode: {Args.CeilMode}";
    }
}
