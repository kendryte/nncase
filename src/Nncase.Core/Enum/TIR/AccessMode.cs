// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.TIR;

/// <summary>
/// TIR access mode.
/// </summary>
public enum AccessMode
{
    /// <summary>
    /// Read only.
    /// </summary>
    Read = 1,

    /// <summary>
    /// Write only.
    /// </summary>
    Write = 2,

    /// <summary>
    /// Read and write.
    /// </summary>
    ReadWrite = 3,
}
