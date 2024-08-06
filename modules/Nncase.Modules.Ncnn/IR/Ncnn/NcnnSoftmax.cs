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
/// Softmax expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class NcnnSoftmax : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(NcnnSoftmax), 0, "input");

    /// <summary>
    /// Gets axis of Ncnn Softmax.
    /// </summary>
    public int Axis { get; }

    /// <inheritdoc/>
    public override string DisplayProperty()
    {
        return $"{Axis}";
    }
}
