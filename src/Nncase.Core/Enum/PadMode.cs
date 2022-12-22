// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase;

/// <summary>
/// Pad mode.
/// </summary>
public enum PadMode : byte
{
    /// <summary>
    /// Constant.
    /// </summary>
    Constant,

    /// <summary>
    /// Reflect.
    /// </summary>
    Reflect,

    /// <summary>
    /// Symmetric.
    /// </summary>
    Symmetric,

    /// <summary>
    /// Edge.
    /// </summary>
    Edge,
}
