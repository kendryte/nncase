// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Text.RegularExpressions;
using Google.OrTools.ConstraintSolver;

namespace Nncase.Schedule.TileTree;

public static class TreeExtensions
{
    private static readonly Regex _rangePattern = new Regex(@"\(\d+..\d+\)", RegexOptions.Compiled);

    public static ITileAbleNode? GetParentTileableNode(this ITreeNode node)
    {
        return node.Parent switch
        {
            ScopeNode s => GetParentTileableNode(s),
            ITileAbleNode s => s,
            _ => null,
        };
    }

    public static ITileAbleNode? GetChildTileableNode(this ITreeNode node)
    {
        return node switch
        {
            ScopeNode s => s.Children.Select(GetChildTileableNode).First(),
            ITileAbleNode s => s,
            _ => null,
        };
    }

    public static IEnumerable<ITileAbleNode> GetChildTileableNodes(this ITreeNode node)
    {
        return node switch
        {
            ScopeNode s => s.Children.Select(GetChildTileableNodes).SelectMany(i => i),
            ITileAbleNode s => new[] { s },
            _ => Array.Empty<ITileAbleNode>(),
        };
    }

    public static string ToSimplifyString(this PropagationBaseObject intExpr)
    {
        var str = intExpr.ToString();
        return _rangePattern.Replace(str, string.Empty);
    }

    public static IR.Expr GetArgument(this IR.Affine.Grid grid, int index)
    {
        return index >= grid.Reads.Length ? grid.Buffers[^1] : grid.Reads[index];
    }
}
