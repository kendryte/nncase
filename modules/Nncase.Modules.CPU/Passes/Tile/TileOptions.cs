// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.Passes.Tile;

/// <summary>
/// TileOptions.
/// </summary>
/// <param name="TargetTileSize">TargetTileSize.</param>
/// <param name="CacheSize">the cache size.</param>
public sealed record TileOptions(int[] TargetTileSize, int CacheSize)
{
    public static TileOptions Default { get; } = new(Array.Empty<int>(), 4 * 1024 * 1024 * 8);
}
