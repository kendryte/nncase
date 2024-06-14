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
public sealed partial class NcnnLayerNorm : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(NcnnLayerNorm), 0, "input");

    /// <summary>
    /// Gets AffineSize of Ncnn LayerNorm.
    /// </summary>
    public int AffineSize { get; }

    /// <summary>
    /// Gets Eps of Ncnn LayerNorm.
    /// </summary>
    public float Eps { get; }

    /// <summary>
    /// Gets Affine of Ncnn LayerNorm.
    /// </summary>
    public int Affine { get; }

    /// <summary>
    /// Gets GammaData of LayerNorm.
    /// </summary>
    public float[] GammaData { get; }

    /// <summary>
    /// Gets BetaData of LayerNorm.
    /// </summary>
    public float[] BetaData { get; }

    /// <inheritdoc/>
    public override string DisplayProperty()
    {
        return $"AffineSize={AffineSize}, eps={Eps}, affine={Affine}";
    }
}
