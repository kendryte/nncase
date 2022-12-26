// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase;

/// <summary>
/// Image resize mode.
/// </summary>
public enum ImageResizeMode : byte
{
    /// <summary>
    /// Bilinear.
    /// </summary>
    Bilinear,

    /// <summary>
    /// Nereast neighbor.
    /// </summary>
    NearestNeighbor,
}

public enum ImageResizeTransformationMode
{
    /// <summary>
    /// HalfPixel.
    /// </summary>
    HalfPixel,

    /// <summary>
    /// PytorchHalfPixel.
    /// </summary>
    PytorchHalfPixel,

    /// <summary>
    /// AlignCorners.
    /// </summary>
    AlignCorners,

    /// <summary>
    /// Asymmetric.
    /// </summary>
    Asymmetric,

    /// <summary>
    /// TFCropAndResize.
    /// </summary>
    TFCropAndResize,
}

public enum ImageResizeNearestMode
{
    /// <summary>
    /// RoundPreferFloor.
    /// </summary>
    RoundPreferFloor,

    /// <summary>
    /// RoundPreferCeil.
    /// </summary>
    RoundPreferCeil,

    /// <summary>
    /// Floor.
    /// </summary>
    Floor,

    /// <summary>
    /// Ceil.
    /// </summary>
    Ceil,
}

public static class ResizeModeHelper
{
    public static ImageResizeMode ParseResizeMode(string mode)
    {
        return mode switch
        {
            "nearest" => ImageResizeMode.NearestNeighbor,
            "linear" => ImageResizeMode.Bilinear,
            _ => throw new NotSupportedException($"Unsupported Resize Mode {mode}"),
        };
    }

    public static string ToString(ImageResizeMode mode)
    {
        return mode switch
        {
            ImageResizeMode.NearestNeighbor => "nearest",
            ImageResizeMode.Bilinear => "linear",
            _ => throw new NotSupportedException($"Unsupported Resize Mode {mode}"),
        };
    }

    public static ImageResizeTransformationMode ParseImageResizeTransformationMode(string mode)
    {
        return mode switch
        {
            "half_pixel" => ImageResizeTransformationMode.HalfPixel,
            "pytorch_half_pixel" => ImageResizeTransformationMode.PytorchHalfPixel,
            "align_corners" => ImageResizeTransformationMode.AlignCorners,
            "asymmetric" => ImageResizeTransformationMode.Asymmetric,
            "tf_crop_and_resize" => ImageResizeTransformationMode.TFCropAndResize,
            _ => throw new NotSupportedException($"Unsupported ResizeTransformationMode Mode {mode}"),
        };
    }

    public static string ToString(ImageResizeTransformationMode mode)
    {
        return mode switch
        {
            ImageResizeTransformationMode.HalfPixel => "half_pixel",
            ImageResizeTransformationMode.PytorchHalfPixel => "pytorch_half_pixel",
            ImageResizeTransformationMode.AlignCorners => "align_corners",
            ImageResizeTransformationMode.Asymmetric => "asymmetric",
            ImageResizeTransformationMode.TFCropAndResize => "tf_crop_and_resize",
            _ => throw new NotSupportedException($"Unsupported ResizeTransformationMode Mode {mode}"),
        };
    }

    public static ImageResizeNearestMode ParseImageResizeNearestMode(string mode)
    {
        return mode switch
        {
            "round_prefer_floor" => ImageResizeNearestMode.RoundPreferFloor,
            "round_prefer_ceil" => ImageResizeNearestMode.RoundPreferCeil,
            "floor" => ImageResizeNearestMode.Floor,
            "ceil" => ImageResizeNearestMode.Ceil,
            _ => throw new NotSupportedException($"Unsupported ResizeTransformationMode Mode {mode}"),
        };
    }

    public static string ToString(ImageResizeNearestMode mode)
    {
        return mode switch
        {
            ImageResizeNearestMode.RoundPreferFloor => "round_prefer_floor",
            ImageResizeNearestMode.RoundPreferCeil => "round_prefer_ceil",
            ImageResizeNearestMode.Floor => "floor",
            ImageResizeNearestMode.Ceil => "ceil",
            _ => throw new NotSupportedException($"Unsupported ResizeTransformationMode Mode {mode}"),
        };
    }
}
