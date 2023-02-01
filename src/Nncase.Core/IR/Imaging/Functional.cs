// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR.Imaging;
using Nncase.IR.NN;
using Nncase.IR.Random;
using Nncase.IR.Tensors;

namespace Nncase.IR.F;

/// <summary>
/// Imaging functional helper.
/// </summary>
public static class Imaging
{
    /// <summary>
    ///  resize image.
    /// </summary>
    public static Call ResizeImage(
        ImageResizeMode resizeMode,
        ImageResizeTransformationMode transformationMode,
        ImageResizeNearestMode nearestMode,
        Expr input,
        Expr roi,
        Expr newSize,
        Expr cubicCoeffA,
        Expr excludeOutside,
        Expr extrapolationValue,
        bool isTFResize = false)
        => new Call(
            new ResizeImage(resizeMode, transformationMode, nearestMode, isTFResize), input, roi, newSize, cubicCoeffA, excludeOutside, extrapolationValue);

    /// <summary>
    /// resize image.
    /// </summary>
    public static Call ResizeImage(
        ImageResizeMode resizeMode,
        Expr input,
        Expr roi,
        Expr newSize,
        ImageResizeTransformationMode tranMode = ImageResizeTransformationMode.Asymmetric,
        ImageResizeNearestMode nearestMode = ImageResizeNearestMode.Floor,
        bool isTFResize = false)
        => ResizeImage(
            resizeMode,
            tranMode,
            nearestMode,
            input,
            roi,
            newSize,
            -0.75f,
            false,
            -0.0f,
            isTFResize);
}
