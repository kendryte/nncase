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
/// Gets expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class NcnnInstanceNorm : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(NcnnInstanceNorm), 0, "input");

    /// <summary>
    /// Gets Channels of Ncnn InstanceNorm.
    /// </summary>
    public int Channels { get; }

    /// <summary>
    /// Gets Eps of Ncnn InstanceNorm.
    /// </summary>
    public float Eps { get; }

    /// <summary>
    /// Gets Affine of Ncnn InstanceNorm.
    /// </summary>
    public int Affine { get; }

    /// <summary>
    /// Gets GammaData of InstanceNorm.
    /// </summary>
    public float[] GammaData { get; }

    /// <summary>
    /// Gets BetaData of InstanceNorm.
    /// </summary>
    public float[] BetaData { get; }

    /// <inheritdoc/>
    public override string DisplayProperty()
    {
        return $"channels={Channels}, eps={Eps}, affine={Affine}";
    }
}
