// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.Schedule;

public static class TileTreeExtensions
{
    public static ITileAbleNode? GetParentTileableNode(this ITreeNode node)
    {
        return node.Parent switch
        {
            ScopeNode s => GetParentTileableNode(s),
            ITileAbleNode s => s,
            _ => null,
        };
    }
}
