// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Dynamic;
using Nncase.IR;

namespace Nncase.TIR.XPU;

public sealed partial class Resize : XPUKernelOp
{
    public static readonly ParameterInfo Input = new(typeof(Resize), 0, "input");

    public static readonly ParameterInfo Output = new(typeof(Resize), 1, "output");

    public IRArray<float> Roi { get; }

    public IRArray<int> NewSize { get; }

    public float CubicCoeffA { get; }

    public int ExcludeOutsideValue { get; }

    public float ExtrapolationValue { get; }

    public ImageResizeMode ResizeMode { get; }

    public ImageResizeTransformationMode TransformationMode { get; }

    public ImageResizeNearestMode NearestMode { get; }

    public bool IsTFResize { get; }

    /// <inheritdoc/>
    public override string DisplayProperty() => $"ImageResizeMode.{ResizeMode}, ImageResizeTransformationMode.{TransformationMode}, ImageResizeNearestMode.{NearestMode}, {IsTFResize}";
}
