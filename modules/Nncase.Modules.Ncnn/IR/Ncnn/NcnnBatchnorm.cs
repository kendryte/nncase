// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.PatternMatch;

namespace Nncase.IR.Ncnn;

/// <summary>
/// BatchNorm expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class NcnnBatchNorm : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(NcnnBatchNorm), 0, "input");

    /// <summary>
    /// Gets channels of Ncnn BatchNorm.
    /// </summary>
    public int Channels { get; }

    /// <summary>
    /// Gets eps.
    /// </summary>
    public float Eps { get; }

    /// <summary>
    /// Gets slopeData of Ncnn BatchNorm.
    /// </summary>
    public float[] SlopeData { get; }

    /// <summary>
    /// Gets meanData of Ncnn BatchNorm.
    /// </summary>
    public float[] MeanData { get; }

    /// <summary>
    /// Gets varData of Ncnn BatchNorm.
    /// </summary>
    public float[] VarData { get; }

    /// <summary>
    /// Gets biasData of Ncnn BatchNorm.
    /// </summary>
    public float[] BiasData { get; }

    /// <inheritdoc/>
    public override string DisplayProperty()
    {
        return $"{Channels},{Eps}";
    }
}
