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
/// <param name="Hierarchy">the hierarchy shapes.</param>
/// <param name="HierarchySizes">each hierarchy ram size.</param>
public sealed record TileOptions(int[] TargetTileSize, int[] Hierarchy, int[] HierarchySizes)
{
    public static TileOptions Default { get; } = new(Array.Empty<int>(), new[] { 1 }, new[] { 64 * (int)MathF.Pow(2, 30) });
}
