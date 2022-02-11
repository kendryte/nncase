// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase;

/// <summary>
/// Image resize mode.
/// </summary>
public enum ImageResizeMode
{
    /// <summary>
    /// Bilinear.
    /// </summary>
    Bilinear,

    /// <summary>
    /// Trilinear.
    /// </summary>
    Trilinear,

    /// <summary>
    /// Nereast neighbor.
    /// </summary>
    NearestNeighbor,
}
