// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.Passes.BufferSchedule;

public class SRAM
{
    public static int SramSizePerBlock { get; } = 2 * 1024 * 1024;

    public static int SramSizePerThread { get; } = SramSizePerBlock / 4;
}
