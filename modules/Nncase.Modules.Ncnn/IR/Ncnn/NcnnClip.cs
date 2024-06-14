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
/// Clip expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class NcnnClip : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(NcnnClip), 0, "input");

    /// <summary>
    /// Gets min of Ncnn Clip.
    /// </summary>
    public float Min { get; }

    /// <summary>
    /// Gets max of Ncnn Clip.
    /// </summary>
    public float Max { get; }

    /// <inheritdoc/>
    public override string DisplayProperty()
    {
        return $"{Min}, {Max}";
    }
}
