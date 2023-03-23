// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.IR.Imaging;

/// <summary>
/// ResizeImage expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class ResizeImage : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(ResizeImage), 0, "input", HasRank(r => r >= 2, "RanK >= 2"));

    /// <summary>
    /// Gets roi.
    /// [axis 0 start,axis 1 end, ... ,axis n start, axis n end].
    /// </summary>
    public static readonly ParameterInfo Roi = new(typeof(ResizeImage), 1, "roi", IsNoneType() | (IsFloat() & HasRank(1)));

    /// <summary>
    /// Gets new_size.
    /// </summary>
    public static readonly ParameterInfo NewSize = new(typeof(ResizeImage), 2, "new_size", HasShape(new[] { 4 }));

    /// <summary>
    /// Gets CubicCoeffA.
    /// </summary>
    public static readonly ParameterInfo CubicCoeffA = new(typeof(ResizeImage), 3, "cubic_coeff_a", IsNoneType() | IsFloatScalar());

    /// <summary>
    /// Gets ExcludeOutside.
    /// </summary>
    public static readonly ParameterInfo ExcludeOutside = new(typeof(ResizeImage), 4, "exclude_outside", IsNoneType() | IsIntegralScalar());

    /// <summary>
    /// Gets ExtrapolationValue.
    /// </summary>
    public static readonly ParameterInfo ExtrapolationValue = new(typeof(ResizeImage), 5, "extrapolation_value", IsNoneType() | IsFloatScalar());

    public ImageResizeMode ResizeMode { get; }

    public ImageResizeTransformationMode TransformationMode { get; }

    public ImageResizeNearestMode NearestMode { get; }

    public bool IsTFResize { get; }

    /// <inheritdoc/>
    public override string DisplayProperty() => $"ImageResizeMode.{ResizeMode}, ImageResizeTransformationMode.{TransformationMode}, ImageResizeNearestMode.{NearestMode}, {IsTFResize}";
}
