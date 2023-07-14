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
/// <param name="ForceFence">ForceFence.</param>
/// <param name="PingPong"> 是否进行ping pong. </param>
/// <param name="PingPongNum">PingPongNum. </param>
/// <param name="ForceMultiLayer"> 对于测试. </param>
/// <param name="MultiWorkers"> 是否开启多线程搜索. </param>
public sealed record TileOptions(int[] TargetTileSize, bool ForceFence, bool PingPong, int PingPongNum, bool ForceMultiLayer, bool MultiWorkers)
{
    public static TileOptions Default { get; } = new(Array.Empty<int>(), false, true, 2, false, true);
}
