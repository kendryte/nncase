// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.ArgsStruct;

/// <summary>
/// Ncnn Pooling arguments.
/// </summary>
public record PoolingArgs
{
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

public record ConvTransposeArgs
{
    public ConvTransposeArgs(Tensor? weightData = null, float[]? biasData = null, int numOutput = default, int kernelW = default, int kernelH = default, int dilationW = default, int dilationH = default, int strideW = default, int strideH = default, int padLeft = default, int padRight = default, int padTop = default, int padBottom = default, int biasTerm = default, int weightDataSize = default, int activationType = default, float[]? activationParams = null, int outputPadRight = default, int outputPadBottom = default, int outputW = default, int outputH = default)
    {
        WeightData = weightData!;
        BiasData = biasData!;
        NumOutput = numOutput;
        KernelW = kernelW;
        KernelH = kernelH;
        DilationW = dilationW;
        DilationH = dilationH;
        StrideW = strideW;
        StrideH = strideH;
        PadLeft = padLeft;
        PadRight = padRight;
        PadTop = padTop;
        PadBottom = padBottom;
        BiasTerm = biasTerm;
        WeightDataSize = weightDataSize;
        ActivationType = activationType;
        ActivationParams = activationParams!;
        OutputPadRight = outputPadRight;
        OutputPadBottom = outputPadBottom;
        OutputW = outputW;
        OutputH = outputH;
    }

    public Tensor WeightData { get; }

    public float[] BiasData { get; }

    public int NumOutput { get; }

    public int KernelW { get; }

    public int KernelH { get; }

    public int DilationW { get; }

    public int DilationH { get; }

    public int StrideW { get; }

    public int StrideH { get; }

    public int PadLeft { get; }

    public int PadRight { get; }

    public int PadTop { get; }

    public int PadBottom { get; }

    public int BiasTerm { get; }

    public int WeightDataSize { get; }

    public int ActivationType { get; }

    public float[] ActivationParams { get; }

    public int OutputPadRight { get; }

    public int OutputPadBottom { get; }

    public int OutputW { get; }

    public int OutputH { get; }

    public override string ToString() => $"{nameof(WeightData)}: {string.Join("_", WeightData.Shape.ToValueArray())}, {nameof(NumOutput)}: {NumOutput}, {nameof(KernelW)}: {KernelW}, {nameof(KernelH)}: {KernelH}, {nameof(DilationW)}: {DilationW}, {nameof(DilationH)}: {DilationH}, {nameof(StrideW)}: {StrideW}, {nameof(StrideH)}: {StrideH}, {nameof(PadLeft)}: {PadLeft}, {nameof(PadRight)}: {PadRight}, {nameof(PadTop)}: {PadTop}, {nameof(PadBottom)}: {PadBottom}, {nameof(BiasTerm)}: {BiasTerm}, {nameof(WeightDataSize)}: {WeightDataSize}, {nameof(OutputPadRight)}: {OutputPadRight}, {nameof(OutputPadBottom)}: {OutputPadBottom}, {nameof(OutputW)}: {OutputW}, {nameof(OutputH)}: {OutputH}";
}

public record ConvArgs
{
    public ConvArgs(float[] weightData, float[] biasData, int numOutput, int kernelW, int kernelH, int dilationW, int dilationH, int strideW, int strideH, int padLeft, int padRight, int padTop, int padBottom, float padValue, int biasTerm, int weightDataSize, int int8ScaleTerm, int activationType, float[] activationParams, int dynamicWeight, int groups)
    {
        WeightData = weightData;
        BiasData = biasData;
        NumOutput = numOutput;
        KernelW = kernelW;
        KernelH = kernelH;
        DilationW = dilationW;
        DilationH = dilationH;
        StrideW = strideW;
        StrideH = strideH;
        PadLeft = padLeft;
        PadRight = padRight;
        PadTop = padTop;
        PadBottom = padBottom;
        PadValue = padValue;
        BiasTerm = biasTerm;
        WeightDataSize = weightDataSize;
        Int8ScaleTerm = int8ScaleTerm;
        ActivationType = activationType;
        ActivationParams = activationParams;
        DynamicWeight = dynamicWeight;
        Groups = groups;
    }

    /// <summary>
    /// Gets input.
    /// </summary>
    public float[] WeightData { get; }

    /// <summary>
    /// Gets BiasData.
    /// </summary>
    public float[] BiasData { get; }

    /// <summary>
    /// Gets NumOutput.
    /// </summary>
    public int NumOutput { get; }

    /// <summary>
    /// Gets KernelW.
    /// </summary>
    public int KernelW { get; }

    /// <summary>
    /// Gets KernelH.
    /// </summary>
    public int KernelH { get; }

    /// <summary>
    /// Gets DilationW.
    /// </summary>
    public int DilationW { get; }

    /// <summary>
    /// Gets DilationH.
    /// </summary>
    public int DilationH { get; }

    /// <summary>
    /// Gets StrideW.
    /// </summary>
    public int StrideW { get; }

    /// <summary>
    /// Gets StrideH.
    /// </summary>
    public int StrideH { get; }

    /// <summary>
    /// Gets PadLeft.
    /// </summary>
    public int PadLeft { get; }

    /// <summary>
    /// Gets PadRight.
    /// </summary>
    public int PadRight { get; }

    /// <summary>
    /// Gets PadTop.
    /// </summary>
    public int PadTop { get; }

    /// <summary>
    /// Gets PadBottom.
    /// </summary>
    public int PadBottom { get; }

    /// <summary>
    /// Gets PadValue.
    /// </summary>
    public float PadValue { get; }

    /// <summary>
    /// Gets BiasTerm.
    /// </summary>
    public int BiasTerm { get; }

    /// <summary>
    /// Gets WeightDataSize.
    /// </summary>
    public int WeightDataSize { get; }

    /// <summary>
    /// Gets Int8ScaleTerm.
    /// </summary>
    public int Int8ScaleTerm { get; }

    /// <summary>
    /// Gets ActivationType.
    /// </summary>
    public int ActivationType { get; }

    /// <summary>
    /// Gets ActivationParams.
    /// </summary>
    public float[] ActivationParams { get; }

    /// <summary>
    /// Gets DynamicWeight.
    /// </summary>
    public int DynamicWeight { get; }

    /// <summary>
    /// Gets Groups.
    /// </summary>
    public int Groups { get; }
}
