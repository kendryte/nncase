// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.ArgsStruct;

/// <summary>
/// Ncnn Pooling arguments.
/// </summary>
public record PoolingArgs
{
    public PoolingArgs(int poolingType, int kernelW, int kernelH, int strideW, int strideH, int padLeft, int padRight,
        int padTop, int padBottom, bool globalPooling, int padMode, bool avgPoolCountIncludePad, bool adaptivePooling,
        int outW, int outH, bool ceilMode)
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

public record ReductionArgs
{
    public ReductionArgs(int opType, int reduceAll, float coeff, long[] axes, int keepdims)
    {
        OpType = opType;
        ReduceAll = reduceAll;
        Coeff = coeff;
        Axes = axes;
        Keepdims = keepdims;
    }

    public int OpType { get; }

    public int ReduceAll { get; }

    public float Coeff { get; }

    public long[] Axes { get; }

    public int Keepdims { get; }
}

public record CropArgs
{
    public CropArgs(int[] starts, int[] ends, int[] axes)
    {
        Woffset = 0;
        Hoffset = 0;
        Doffset = 0;
        Coffset = 0;
        Outw = 0;
        Outh = 0;
        Outd = 0;
        Outc = 0;
        Woffset2 = 0;
        Hoffset2 = 0;
        Doffset2 = 0;
        Coffset2 = 0;
        Starts = starts;
        Ends = ends;
        Axes = axes;
    }

    public CropArgs(int woffset, int hoffset, int doffset, int coffset, int outw, int outh, int outd, int outc, int woffset2, int hoffset2, int doffset2, int coffset2)
    {
        Woffset = woffset;
        Hoffset = hoffset;
        Doffset = doffset;
        Coffset = coffset;
        Outw = outw;
        Outh = outh;
        Outd = outd;
        Outc = outc;
        Woffset2 = woffset2;
        Hoffset2 = hoffset2;
        Doffset2 = doffset2;
        Coffset2 = coffset2;
        Starts = null;
        Ends = null;
        Axes = null;
    }

    public int Woffset { get; }

    public int Hoffset { get; }

    public int Doffset { get; }

    public int Coffset { get; }

    public int Outw { get; }

    public int Outh { get; }

    public int Outd { get; }

    public int Outc { get; }

    public int Woffset2 { get; }

    public int Hoffset2 { get; }

    public int Doffset2 { get; }

    public int Coffset2 { get; }

    public int[]? Starts { get; }

    public int[]? Ends { get; }

    public int[]? Axes { get; }
}
