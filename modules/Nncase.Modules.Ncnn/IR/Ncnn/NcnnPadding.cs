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
public sealed partial class NcnnPadding : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(NcnnPadding), 0, "input");

    /// <summary>
    /// Gets Top of Ncnn Padding.
    /// </summary>
    public int Top { get; }

    /// <summary>
    /// Gets Bottom of Ncnn Padding.
    /// </summary>
    public int Bottom { get; }

    /// <summary>
    /// Gets Left of Ncnn Padding.
    /// </summary>
    public int Left { get; }

    /// <summary>
    /// Gets Right of Ncnn Padding.
    /// </summary>
    public int Right { get; }

    /// <summary>
    /// Gets Type of Ncnn Padding.
    /// </summary>
    public int Type { get; }

    /// <summary>
    /// Gets Value of Ncnn Padding.
    /// </summary>
    public float Value { get; }

    // /// <summary>
    // /// Gets PerChannelPadSize of Ncnn Padding. Do not need in onnx model.
    // /// </summary>
    // public int PerChannelPadSize { get; }

    /// <summary>
    /// Gets Front of Ncnn Padding.
    /// </summary>
    public int Front { get; }

    /// <summary>
    /// Gets Behind of Ncnn Padding.
    /// </summary>
    public int Behind { get; }

    /// <inheritdoc/>
    public override string DisplayProperty()
    {
        return $"Top: {Top}, Bottom: {Bottom}, Left: {Left}, Right: {Right}, Type: {Type}, Value: {Value}, Front: {Front}, Behind: {Behind}";
    }
}
