// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections;
using System.Diagnostics.CodeAnalysis;
using System.Reactive;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.Schedule.MonteCarloTreeSearch;
using QuikGraph;
using QuikGraph.Algorithms;
using QuikGraph.Algorithms.ShortestPath;
using QuikGraph.Graphviz;

namespace Nncase.Schedule.TileGraph;

public sealed class MCTState : IEnvironmentState<MergePoint>
{
    private static int _globl_count;

    private readonly List<MergePoint> _mergePoints = new();

    private readonly List<int> _legalIndex = new();

    private readonly Dictionary<TieredTileGraph, Expr> _resultMemo = new();

    private readonly int _count;

    public MCTState(TieredTileGraph graph, string moduleKind, string itemNumber, Dictionary<TileNode, GraphTiler.TiledFunc> tilingMemo, ICpuTargetOptions targetOptions)
    {
        Graph = graph;
        ModuleKind = moduleKind;
        ItemNumber = itemNumber;
        TilingMemo = tilingMemo;
        TargetOptions = targetOptions;
        _mergePoints.AddRange(graph.GetMergePoints());
        _legalIndex.AddRange(Enumerable.Range(0, _mergePoints.Count));
        _count = _globl_count++;
    }

    public long ObjectValue { get; private set; }

    public TieredTileGraph Graph { get; }

    public string ModuleKind { get; }

    public string ItemNumber { get; }

    public Dictionary<TileNode, GraphTiler.TiledFunc> TilingMemo { get; }

    public ICpuTargetOptions TargetOptions { get; }

    public MergePoint GetNextAction(int index)
    {
        var legalIndex = _legalIndex[index];
        _legalIndex.Remove(legalIndex);
        return _mergePoints[legalIndex];
    }

    public int LegalActions()
    {
        return _legalIndex.Count;
    }

    public IEnvironmentState<MergePoint> PerformAction(MergePoint mergePoint)
    {
        var newGraph = Graph.Clone();
        newGraph.Merge(mergePoint);
        return new MCTState(newGraph, ModuleKind, ItemNumber, TilingMemo, TargetOptions);
    }

    public double RollOut()
    {
        if (ObjectValue == 0)
        {
            using var scope = new Diagnostics.DumpScope($"MCTS{_count}");
            var res = GraphTiler.SolveRootGraph(Graph, ModuleKind, ItemNumber, TilingMemo, TargetOptions);
            ObjectValue = res.ObjectValue;
            foreach (var item in res.ResultMemo)
            {
                _resultMemo.Add(item.Key, item.Value);
            }
        }

        return ObjectValue;
    }
}

public sealed class MCTNode : SearchNode<MergePoint>
{
    public MCTNode(IEnvironmentState<MergePoint> state)
        : base(state)
    {
    }

    public MCTNode(SearchNode<MergePoint> parent, IEnvironmentState<MergePoint> state)
        : base(parent, state)
    {
    }

    public override void Update(double reward)
    {
        if (QualityValue > reward)
        {
            QualityValue = reward;
        }

        VisitTimes += 1;

        if (Parent is not null)
        {
            Parent.Update(reward);
        }
    }
}

public sealed class MCTSearcher : Searcher<MergePoint>
{
    private readonly Random _random = new Random(1010);

    public MCTSearcher()
    {
        BestObjectValue = double.PositiveInfinity;
    }

    public double BestObjectValue { get; private set; }

    public SearchNode<MergePoint> UCBSelectChild(SearchNode<MergePoint> node)
    {
        double coef = Math.Sqrt(2);
        double temp = 0.5;
        var ucbs = node.Children.Select(c => (-c.QualityValue / BestObjectValue) + (coef * Math.Sqrt(Math.Log(node.VisitTimes) / c.VisitTimes))).ToArray();
        var ucbs_exp = ucbs.Select(ucb => Math.Exp(ucb / temp)).ToArray();
        var sum = ucbs_exp.Sum();
        var probs = ucbs_exp.Select(e => (int)(e / sum * 30)).ToArray(); // conver ucb as prob
        var candidates = probs.Select((p, i) => Enumerable.Repeat(i, p).ToArray()).SelectMany(i => i).ToArray();
        return node.Children[_random.GetItems(candidates, 1)[0]];
    }

    public override bool Selection(SearchNode<MergePoint> node, out SearchNode<MergePoint> selected)
    {
        while (node.State.LegalActions() == 0 && node.Children.Count > 0)
        {
            node = UCBSelectChild(node);
        }

        selected = node;
        return true;
    }

    public override SearchNode<MergePoint> Expand(SearchNode<MergePoint> node)
    {
        if (node.VisitTimes != 0 && node.State.LegalActions() > 0)
        {
            var index = _random.Next(node.State.LegalActions());
            var action = node.State.GetNextAction(index);
            var state = node.State.PerformAction(action);
            return new MCTNode(node, state);
        }

        return node;
    }

    public override double Simulation(SearchNode<MergePoint> node)
    {
        double value = node.State.RollOut();
        if (value < BestObjectValue)
        {
            BestObjectValue = value;
        }

        return value;
    }

    public override void BackPropagate(SearchNode<MergePoint> node, double reward)
    {
        node.Update(reward);
    }
}
