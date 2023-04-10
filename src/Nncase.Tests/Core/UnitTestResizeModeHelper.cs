// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Microsoft.Toolkit.HighPerformance.Helpers;
using Nncase;
using Nncase.IR;
using Xunit;

namespace Nncase.Tests.CoreTest;

public sealed class UnitTestResizeModeHelper
{
    [Fact]
    public void TestParseResizeMode()
    {
        Assert.Equal(ImageResizeMode.NearestNeighbor, ResizeModeHelper.ParseResizeMode("nearest"));
        Assert.Equal(ImageResizeMode.Bilinear, ResizeModeHelper.ParseResizeMode("linear"));
        Assert.Throws<NotSupportedException>(() => ResizeModeHelper.ParseResizeMode(string.Empty));
    }

    [Fact]
    public void TestToString()
    {
        Assert.Equal("nearest", ResizeModeHelper.ToString(ImageResizeMode.NearestNeighbor));
        Assert.Equal("linear", ResizeModeHelper.ToString(ImageResizeMode.Bilinear));

        Assert.Equal("half_pixel", ResizeModeHelper.ToString(ImageResizeTransformationMode.HalfPixel));
        Assert.Equal("pytorch_half_pixel", ResizeModeHelper.ToString(ImageResizeTransformationMode.PytorchHalfPixel));
        Assert.Equal("align_corners", ResizeModeHelper.ToString(ImageResizeTransformationMode.AlignCorners));
        Assert.Equal("asymmetric", ResizeModeHelper.ToString(ImageResizeTransformationMode.Asymmetric));
        Assert.Equal("tf_crop_and_resize", ResizeModeHelper.ToString(ImageResizeTransformationMode.TFCropAndResize));

        Assert.Equal("round_prefer_floor", ResizeModeHelper.ToString(ImageResizeNearestMode.RoundPreferFloor));
        Assert.Equal("round_prefer_ceil", ResizeModeHelper.ToString(ImageResizeNearestMode.RoundPreferCeil));
        Assert.Equal("floor", ResizeModeHelper.ToString(ImageResizeNearestMode.Floor));
        Assert.Equal("ceil", ResizeModeHelper.ToString(ImageResizeNearestMode.Ceil));
    }

    [Fact]
    public void TestImageResizeTransformationMode()
    {
        Assert.Equal(ImageResizeTransformationMode.HalfPixel, ResizeModeHelper.ParseImageResizeTransformationMode("half_pixel"));
        Assert.Equal(ImageResizeTransformationMode.PytorchHalfPixel, ResizeModeHelper.ParseImageResizeTransformationMode("pytorch_half_pixel"));
        Assert.Equal(ImageResizeTransformationMode.AlignCorners, ResizeModeHelper.ParseImageResizeTransformationMode("align_corners"));
        Assert.Equal(ImageResizeTransformationMode.Asymmetric, ResizeModeHelper.ParseImageResizeTransformationMode("asymmetric"));
        Assert.Equal(ImageResizeTransformationMode.TFCropAndResize, ResizeModeHelper.ParseImageResizeTransformationMode("tf_crop_and_resize"));
        Assert.Throws<NotSupportedException>(() => ResizeModeHelper.ParseResizeMode(string.Empty));
    }

    [Fact]
    public void TestParseImageResizeNearestMode()
    {
        Assert.Equal(ImageResizeNearestMode.RoundPreferFloor, ResizeModeHelper.ParseImageResizeNearestMode("round_prefer_floor"));
        Assert.Equal(ImageResizeNearestMode.RoundPreferCeil, ResizeModeHelper.ParseImageResizeNearestMode("round_prefer_ceil"));
        Assert.Equal(ImageResizeNearestMode.Floor, ResizeModeHelper.ParseImageResizeNearestMode("floor"));
        Assert.Equal(ImageResizeNearestMode.Ceil, ResizeModeHelper.ParseImageResizeNearestMode("ceil"));
        Assert.Throws<NotSupportedException>(() => ResizeModeHelper.ParseImageResizeNearestMode(string.Empty));
    }
}
