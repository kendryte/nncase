// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.TIR;

/// <summary>
/// Buffer mode.
/// </summary>
public enum BufferMode
{
    /// <summary>
    /// Default.
    /// </summary>
    Default,

    /// <summary>
    /// Maps buffer[i][j][k] -> buffer[i][0][k] if dimension i's shape equals 1.
    /// </summary>
    AutoBroadcast,
}
