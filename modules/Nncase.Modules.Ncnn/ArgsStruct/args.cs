// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.ArgsStruct;
/// <summary>
/// Ncnn Pooling arguments.
/// </summary>
public record PoolingArgs
{
    public PoolingArgs()
    {
    }

    public PoolingArgs(int poolingType, int kernelW, int kernelH, int strideW, int strideH, int padLeft, int padRight, int padTop, int padBottom, bool globalPooling, int padMode, bool avgPoolCountIncludePad, bool adaptivePooling, int outW, int outH, bool ceilMode)
    {
        PoolingType = poolingType;
        KernelH = kernelH;
        KernelW = kernelW;
        StrideH = strideH;
        StrideW = strideW;
        PadLeft = padLeft;
        PadRight = padRight;
        PadTop = padTop;
        PadBottom = padBottom;
        GlobalPooling = globalPooling;
        PadMode = padMode;
        AvgPoolCountIncludePad = avgPoolCountIncludePad;
        AdaptivePooling = adaptivePooling;
        OutW = outW;
        OutH = outH;
        CeilMode = ceilMode;
    }

    public int PoolingType { get; }

    public int KernelW { get; }

    public int KernelH { get; }

    public int StrideW { get; }

    public int StrideH { get; }

    public int PadLeft { get; }

    public int PadRight { get; }

    public int PadTop { get; }

    public int PadBottom { get; }

    public bool GlobalPooling { get; }

    public int PadMode { get; }

    public bool AvgPoolCountIncludePad { get; }

    public bool AdaptivePooling { get; }

    public int OutW { get; }

    public int OutH { get; }

    public bool CeilMode { get; }
}
