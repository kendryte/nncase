// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Text.RegularExpressions;
using GiGraph.Dot.Extensions;
using Google.OrTools.ConstraintSolver;

namespace Nncase.Schedule.TileTree;

public static class TreeExtensions
{
    private static readonly Regex _rangePattern = new Regex(@"\(\d+..\d+\)", RegexOptions.Compiled);

    public static void Walk(this ITreeNode node, Action<ITreeNode> func, bool preOrder = true)
    {
        var functor = new TreeFunctor(func, preOrder);
        node.Accept(functor, default);
        return;
    }

    public static T Root<T>(this ITreeNode node)
        where T : ITreeNode
    {
        List<ITreeNode> stack = new();
        ITreeNode? cur = node;
        while (cur is ITreeNode)
        {
            stack.Add(cur);
            cur = cur.Parent;
        }

        return stack.Reverse<ITreeNode>().OfType<T>().First();
    }

    public static ITreeNode Clone(this ITreeNode node)
    {
        var cloner = new TreeCloner();
        return node.Accept(cloner, default);
    }

    public static void Dump(this ITreeNode tree, string name)
    {
        using (var stream = Diagnostics.DumpScope.Current.OpenFile($"{name}.dot"))
        {
            using var writer = new StreamWriter(stream);
            var printer = new TreePrinter();
            tree.Accept(printer, TreePrinter.Context.Default);
            printer.Graph.Build(writer);
        }
    }

    public static bool Merge(this ITreeNode tree, int opConsumer, int opProducer, int level)
    {
        var merger = new TreeMerger(opConsumer, opProducer, level);
        return tree.Accept(merger, default);
    }

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
