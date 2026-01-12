// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase;

public enum GridSampleAlignCorners : byte
{
    None,
    AlignCorners,
}

public enum GridSampleMode : byte
{
    Bilinear,
    NearestNeighbor,
    Cubic,
}

public enum GridSamplePaddingMode
{
    Zeros,
    Border,
    Reflection,
}

public static class GridSampleModeHelper
{
    public static GridSampleAlignCorners ParseGridSampleAlignCorners(long mode)
    {
        return mode switch
        {
            0 => GridSampleAlignCorners.None,
            1 => GridSampleAlignCorners.AlignCorners,
            _ => throw new NotSupportedException($"Unsupported GridSampleAlignCorners Mode {mode}"),
        };
    }

    public static long ToInt(GridSampleAlignCorners mode)
    {
        return mode switch
        {
            GridSampleAlignCorners.None => 0,
            GridSampleAlignCorners.AlignCorners => 1,
            _ => throw new NotSupportedException($"Unsupported GridSampleAlignCorners Mode {mode}"),
        };
    }

    public static GridSampleMode ParseGridSampleMode(string mode)
    {
        return mode switch
        {
            "nearest" => GridSampleMode.NearestNeighbor,
            "bilinear" => GridSampleMode.Bilinear,
            "cubic" => GridSampleMode.Cubic,
            _ => throw new NotSupportedException($"Unsupported GridSample Mode {mode}"),
        };
    }

    public static string ToString(GridSampleMode mode)
    {
        return mode switch
        {
            GridSampleMode.NearestNeighbor => "nearest",
            GridSampleMode.Bilinear => "bilinear",
            GridSampleMode.Cubic => "cubic",
            _ => throw new NotSupportedException($"Unsupported GridSample Mode {mode}"),
        };
    }

    public static GridSamplePaddingMode ParseGridSamplePaddingMode(string mode)
    {
        return mode switch
        {
            "zeros" => GridSamplePaddingMode.Zeros,
            "border" => GridSamplePaddingMode.Border,
            "reflection" => GridSamplePaddingMode.Reflection,
            _ => throw new NotSupportedException($"Unsupported GridSamplePaddingMode Mode {mode}"),
        };
    }

    public static string ToString(GridSamplePaddingMode mode)
    {
        return mode switch
        {
            GridSamplePaddingMode.Zeros => "zeros",
            GridSamplePaddingMode.Border => "border",
            GridSamplePaddingMode.Reflection => "reflection",
            _ => throw new NotSupportedException($"Unsupported GridSamplePaddingMode Mode {mode}"),
        };
    }
}
