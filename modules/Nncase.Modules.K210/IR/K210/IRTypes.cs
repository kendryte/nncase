// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.IR.K210;

/// <summary>
/// KPU filter type.
/// </summary>
public enum KPUFilterType
{
    Filter_1x1 = 0,
    Filter_3x3 = 1,
}

/// <summary>
/// KPU pool type.
/// </summary>
public enum KPUPoolType
{
    Bypass = 0,
    Max_2_S2 = 1,
    Mean_2_S2 = 2,
    Max_4_S4 = 3,
    Mean_4_S4 = 4,
    LeftTop_2_S2 = 5,
    RightTop_2_S2 = 6,
    LeftTop_4_S4 = 7,
    Mean_2_S1 = 8,
    Max_2_S1 = 9,
}

public record struct KPUBatchNormSegment
{
    public int Mul { get; set; }

    public int Shift { get; set; }

    public int Add { get; set; }
}

public record struct KPUActivationSegment
{
    public long StartX { get; set; }

    public int Mul { get; set; }

    public int Shift { get; set; }

    public int Add { get; set; }
}

public record struct FakeKPUActivationSegment
{
    public float StartX { get; set; }

    public float Mul { get; set; }

    public float Add { get; set; }
}

public record struct Kpu_conv2d_quant_args
{
    public int ArgX;
    public int ShiftX;
    public int ArgW;
    public int ShiftW;
    public int ArgAdd;
}

/// <summary>
/// KPU constants.
/// </summary>
public static class KPUConstants
{
    /// <summary>
    /// KPU RAM size.
    /// </summary>
    public const int RAMSize = 2 * 1024 * 1024; // 2MB

    /// <summary>
    /// BN output bits.
    /// </summary>
    public const int BNOutBits = 36;
}

public class KPUActivationParameters
{
    public KPUActivationSegment[] Segments { get; } = new KPUActivationSegment[16];
}

public class KPUBatchNormParameters
{
    public KPUBatchNormSegment[] Segments { get; } = Array.Empty<KPUBatchNormSegment>();
}

public record class FakeKPUActivationParameters
{
    public FakeKPUActivationSegment[] Segments { get; set; } = Array.Empty<FakeKPUActivationSegment>();

    public ValueRange<float> Clamp { get; set; }
}
