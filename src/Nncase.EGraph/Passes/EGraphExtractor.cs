// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Google.OrTools.Sat;
using Nncase.CostModel;
using Nncase.Diagnostics;
using Nncase.IR;

namespace Nncase.Passes;

public delegate void EGraphExtractConstrains(CpModel model, IReadOnlyDictionary<ENode, BoolVar> vars);

internal class EGraphExtractor
{
    private readonly EGraphCostModel _costModel;

    public EGraphExtractor(EGraphCostModel costModel)
    {
        _costModel = costModel;
    }

    public Expr Extract(EClass root, IEGraph eGraph, EGraphExtractConstrains[] constrains)
    {
        var cpmodel = new CpModel();

        // 0. create bool var for all enode.
        var varMemo = new Dictionary<ENode, BoolVar>();
        foreach (var cls in eGraph.Classes)
        {
            foreach (var (e, i) in cls.Nodes.Select((e, i) => (e, i)))
            {
                varMemo.Add(e, cpmodel.NewBoolVar($"{cls.Id}_{i}"));
            }
        }

        // 1. must pick one in root enode.
        cpmodel.AddBoolOr(root.Nodes.Select(n => varMemo[n]).ToArray());

        // 2. when pick node, must pick one child node.
        foreach (var n in eGraph.Nodes)
        {
            var ns = new[] { varMemo[n].Not() };
            foreach (var child in n.Children)
            {
                cpmodel.AddBoolOr(ns.Concat(child.Nodes.Select(cn => varMemo[cn])));
            }
        }

        // 3. no cycle
        {
            var hgraph = ToHyperGraph(root);
            var class_cycles = FindCycles(hgraph);
            foreach (var cycle in class_cycles)
            {
                if (cycle.Count == 1)
                {
                    foreach (var n in cycle[0].Nodes)
                    {
                        if (n.Children.Contains(cycle[0]))
                        {
                            cpmodel.AddAssumption(varMemo[n].Not());
                        }
                    }
                }
                else
                {
                    // build clauses.
                    var clauses = new List<List<BoolVar>>();
                    for (int i = 0; i < cycle.Count; i++)
                    {
                        var next_hop = (i + 1) % cycle.Count;
                        var u = hgraph.Edges(cycle[i])!;
                        var v = u[cycle[next_hop]];
                        clauses.Add(v.Select(n => varMemo[n]).ToList());
                    }

                    var clauseMemo = new Dictionary<int, BoolVar>();
                    for (int i = 0; i < clauses.Count; i++)
                    {
                        var clause = clauses[i];
                        if (clause.Count > 1)
                        {
                            var tmpV = cpmodel.NewBoolVar(string.Empty);
                            cpmodel.AddBoolAnd(clause.Select(c => c.Not())).OnlyEnforceIf(tmpV);
                            cpmodel.AddBoolOr(clause).OnlyEnforceIf(tmpV.Not());
                            clauseMemo.Add(i, tmpV);
                        }
                    }

                    cpmodel.AddBoolOr(clauses.Select((c, i) => (c, i)).Select(p => p.c.Count == 1 ? p.c[0].Not() : clauseMemo[p.i]));
                }
            }
        }

        foreach (var constrain in constrains)
        {
            constrain(cpmodel, varMemo);
        }

        // 3. add pick weights for all enode.
        cpmodel.Minimize(LinearExpr.WeightedSum(eGraph.Nodes.Select(n => varMemo[n]), eGraph.Nodes.Select(n => checked((long)_costModel[n].Score))));

        if (cpmodel.Validate().Any())
        {
            throw new InvalidDataException("the sat model invalid: " + cpmodel.Validate());
        }

        var solver = new CpSolver();
        int max_time = 120;
        if (System.Environment.GetEnvironmentVariable("SOLVE_MAX_TIME") is string s_solve_max_time)
        {
            try
            {
                var solve_max_time = int.Parse(s_solve_max_time);
                max_time = solve_max_time;
            }
            catch (System.Exception)
            {
            }
        }

        int processorCount = Math.Max(System.Environment.ProcessorCount / 2, 1);
        if (System.Environment.GetEnvironmentVariable("SOLVE_PROCESSOR_COUNT") is string s_solve_processor_count)
        {
            try
            {
                var solve_processor_count = int.Parse(s_solve_processor_count);
                processorCount = solve_processor_count;
            }
            catch (System.Exception)
            {
            }
        }

        solver.StringParameters = $"max_time_in_seconds:{max_time},num_workers:{processorCount}";

        var enableDump = DumpScope.Current.IsEnabled(DumpFlags.EGraphCost);
        CpSolverStatus status;
        using (var dumpStream = enableDump ? DumpScope.Current.OpenFile("Costs/Solve.txt") : Stream.Null)
        {
            using var writer = new StreamWriter(dumpStream);
            var cb = new PrintCostCallBack(varMemo, _costModel, writer, enableDump);
            status = solver.Solve(cpmodel, cb);
            writer.WriteLine($"Status : {status}");
            dumpStream.Flush();
        }

        if (status is not (CpSolverStatus.Optimal or CpSolverStatus.Feasible))
        {
            throw new InvalidProgramException("SatExtract Failed!");
        }

        var picks = eGraph.Nodes.ToDictionary(e => e, e => solver.BooleanValue(varMemo[e]));
        using (var dumpStream = enableDump ? DumpScope.Current.OpenFile("Costs/Pick.dot") : Stream.Null)
        {
            EGraphPrinter.DumpEgraphAsDot(eGraph, _costModel, picks, root.Find(), dumpStream);
        }

        return new SatExprBuildVisitor(picks).Visit(root);
    }

    private static HyperGraph ToHyperGraph(EClass root)
    {
        var hgraph = new HyperGraph();
        var visited = new HashSet<EClass>();
        var queue = new Queue<EClass>();
        queue.Enqueue(root);
        visited.Add(root);
        while (queue.Any())
        {
            var front = queue.Dequeue();
            foreach (var node in front.Nodes)
            {
                foreach (var ch in node.Children)
                {
                    var canonical = ch;
                    hgraph.Connect(front, canonical, node);
                    if (!visited.Contains(canonical))
                    {
                        visited.Add(canonical);
                        queue.Enqueue(canonical);
                    }
                }
            }
        }

        return hgraph;
    }

    private static List<List<EClass>> FindCycles(HyperGraph hgraph)
    {
        var edges = hgraph.AdjacencyEdges();
        var circuits = new List<List<EClass>>();
        foreach (var n in hgraph.Nodes())
        {
            if (hgraph.Neighbors(n).Contains(n))
            {
                circuits.Add(new() { n });
            }
        }

        var stack = new List<int>();
        bool[] blocked = new bool[hgraph.NumEdges()];
        var blockMap = new Dictionary<int, Dictionary<int, bool>>();
        List<List<int>> adjList;
        int s = 0;

        void Unblock(int u)
        {
            blocked[u] = false;
            if (blockMap.ContainsKey(u))
            {
                foreach (var w in blockMap[u].Keys)
                {
                    blockMap[u].Remove(w);
                    if (blocked[w])
                    {
                        Unblock(w);
                    }
                }
            }
        }

        bool Circuit(int v)
        {
            bool found = false;

            stack.Add(v);
            blocked[v] = true;

            for (int i = 0; i < adjList[v].Count; i++)
            {
                int w = adjList[v][i];
                if (w == s)
                {
                    Output(s, stack);
                    found = true;
                }
                else if (!blocked[w])
                {
                    found = Circuit(w);
                }
            }

            if (found)
            {
                Unblock(v);
            }
            else
            {
                for (int i = 0; i < adjList[v].Count; i++)
                {
                    int w = adjList[v][i];
                    if (!blockMap.ContainsKey(w))
                    {
                        blockMap[w] = new Dictionary<int, bool>();
                    }

                    blockMap[w][w] = true;
                }
            }

            stack.RemoveAt(stack.Count - 1);
            return found;
        }

        void Output(int start, List<int> stack)
        {
            // var cycle = new List<int>(stack);
            // {
            //     start,
            // };
            circuits.Add(stack.Select(hgraph.GetIdByNode).ToList());
        }

        void Subgraph(int minId)
        {
            for (int i = 0; i < edges!.Count; i++)
            {
                if (i < minId || edges[i] == null)
                {
                    edges[i] = new List<int>();
                }

                edges[i] = edges[i].Where(j => j >= minId).ToList();
            }
        }

        (int LeastVertex, List<List<int>> AdjList) AdjacencyStructureSCC(int from)
        {
            Subgraph(from);
            List<List<int>> g = edges;

            var (components, _) = StronglyConnectedComponents(edges);

            var ccs = components.Where(scc => scc.Count > 1).ToList();

            int leastVertex = int.MaxValue;
            int leastVertexComponent = -1;
            for (int i = 0; i < ccs.Count; i++)
            {
                for (int j = 0; j < ccs[i].Count; j++)
                {
                    if (ccs[i][j] < leastVertex)
                    {
                        leastVertex = ccs[i][j];
                        leastVertexComponent = i;
                    }
                }
            }

            var cc = leastVertexComponent >= 0 ? ccs[leastVertexComponent] : null;

            if (cc == null)
            {
                return (-1, new());
            }

            var adjList = edges.Select((l, index) =>
            {
                if (cc.IndexOf(index) == -1)
                {
                    return new();
                }

                return l.Where(i => cc.IndexOf(i) != -1).ToList();
            }).ToList();

            return (leastVertex, adjList);
        }

        while (s < edges.Count)
        {
            var (leastVertex, leastAdjList) = AdjacencyStructureSCC(s);
            s = leastVertex;
            adjList = leastAdjList;

            if (adjList.Any())
            {
                for (int i = 0; i < adjList.Count; i++)
                {
                    for (int j = 0; j < adjList[i].Count; j++)
                    {
                        int vertexId = adjList[i][j];
                        blocked[vertexId] = false;
                        if (!blockMap.ContainsKey(vertexId))
                        {
                            blockMap[vertexId] = new Dictionary<int, bool>();
                        }
                    }
                }

                Circuit(s);
                s++;
            }
            else
            {
                s = edges.Count;
            }
        }

        return circuits;
    }

    private static (List<List<int>> Components, List<List<int>> AdjacencyList) StronglyConnectedComponents(List<List<int>> adjList)
    {
        int numVertices = adjList.Count;
        int[] index = new int[numVertices];
        int[] lowValue = new int[numVertices];
        bool[] active = new bool[numVertices];
        int[] child = new int[numVertices];
        int[] scc = new int[numVertices];
        var sccLinks = new List<int>[numVertices];

        // Initialize tables
        for (int i = 0; i < numVertices; ++i)
        {
            index[i] = -1;
            lowValue[i] = 0;
            active[i] = false;
            child[i] = 0;
            scc[i] = -1;
            sccLinks[i] = new List<int>();
        }

        int count = 0;
        var components = new List<List<int>>();
        var sccAdjList = new List<List<int>>();

        void StrongConnect(int v)
        {
            var s = new Stack<int>();
            var t = new Stack<int>();
            s.Push(v);
            t.Push(v);
            index[v] = lowValue[v] = count;
            active[v] = true;
            count++;

            while (t.Count > 0)
            {
                v = t.Peek();
                var e = adjList[v];
                if (child[v] < e.Count)
                {
                    int i;
                    for (i = child[v]; i < e.Count; ++i)
                    {
                        int u = e[i];
                        if (index[u] < 0)
                        {
                            index[u] = lowValue[u] = count;
                            active[u] = true;
                            count++;
                            s.Push(u);
                            t.Push(u);
                            break;
                        }
                        else if (active[u])
                        {
                            lowValue[v] = Math.Min(lowValue[v], lowValue[u]);
                        }

                        if (scc[u] >= 0)
                        {
                            sccLinks[v].Add(scc[u]);
                        }
                    }

                    child[v] = i;
                }
                else
                {
                    if (lowValue[v] == index[v])
                    {
                        var component = new List<int>();
                        var links = Enumerable.Range(0, s.Count).Select(i => new List<int>()).ToArray();
                        int linkCount = 0;
                        for (int i = s.Count - 1; i >= 0; --i)
                        {
                            int w = s.Pop();
                            active[w] = false;
                            component.Add(w);
                            links[i] = sccLinks[w];
                            linkCount += sccLinks[w].Count;
                            scc[w] = components.Count;
                            if (w == v)
                            {
                                break;
                            }
                        }

                        components.Add(component);
                        var allLinks = new List<int>(linkCount);
                        for (int i = 0; i < links.Length; i++)
                        {
                            for (int j = 0; j < links[i].Count; j++)
                            {
                                allLinks.Add(links[i][j]);
                            }
                        }

                        sccAdjList.Add(allLinks);
                    }

                    t.Pop();
                }
            }
        }

        // Run strong connect starting from each vertex
        for (int i = 0; i < numVertices; ++i)
        {
            if (index[i] < 0)
            {
                StrongConnect(i);
            }
        }

        // Compact sccAdjList
        for (int i = 0; i < sccAdjList.Count; i++)
        {
            var e = sccAdjList[i];
            if (e.Count == 0)
            {
                continue;
            }

            e.Sort();
            var newE = new List<int> { e[0] };
            for (int j = 1; j < e.Count; j++)
            {
                if (e[j] != e[j - 1])
                {
                    newE.Add(e[j]);
                }
            }

            sccAdjList[i] = newE;
        }

        return (components, sccAdjList);
    }
}

internal sealed class PrintCostCallBack : CpSolverSolutionCallback
{
    private readonly IReadOnlyDictionary<ENode, BoolVar> _vars;
    private readonly EGraphCostModel _costModel;
    private readonly StreamWriter _dumpWriter;
    private readonly bool _enableDump;
    private int _count;

    public PrintCostCallBack(IReadOnlyDictionary<ENode, BoolVar> vars, EGraphCostModel costModel, StreamWriter writer, bool enableDump)
    {
        _vars = vars;
        _costModel = costModel;
        _dumpWriter = writer;
        _enableDump = enableDump;
    }

    public override void OnSolutionCallback()
    {
        if (_enableDump)
        {
            var cost = Cost.Zero;
            foreach (var (n, v) in _vars)
            {
                if (_costModel[n] != Cost.Zero && BooleanValue(v))
                {
                    cost += _costModel[n];
                }
            }

            _dumpWriter.WriteLine($"Solution {_count++} @ {WallTime()}:");
            _dumpWriter.WriteLine(cost.ToString());
            _dumpWriter.Flush();
        }
    }
}

internal sealed class SatExprBuildVisitor
{
    private readonly IReadOnlyDictionary<ENode, bool> _pick;
    private readonly Dictionary<EClass, Expr> _memo;

    public SatExprBuildVisitor(IReadOnlyDictionary<ENode, bool> pick)
    {
        _pick = pick;
        _memo = new();
    }

    public Expr Visit(EClass root)
    {
        Expr? expr;
        if (_memo.TryGetValue(root, out expr))
        {
            return expr;
        }

        var enodes = root.Nodes.Where(n => _pick[n]).ToArray();
        if (enodes.Length != 1)
        {
            throw new InvalidProgramException("the one eclass only can pick one enode!");
        }

        var enode = enodes[0];
        var children = enode.Children.Select(e => Visit(e)).ToArray();

        switch (enode.Expr)
        {
            case Var or TensorConst or TupleConst or Op or Fusion or None:
                expr = enode.Expr;
                break;
            case Function func:
                expr = children.Length == 0 ? func : func.With(body: children[0], parameters: children[1..].OfType<Var>().ToArray());
                break;
            case Call call:
                expr = call.With(target: children[0], arguments: children[1..], call.Metadata);
                break;
            case IR.Tuple tp:
                expr = tp.With(fields: children);
                break;
            case Marker mk:
                expr = mk.With(target: children[0], attribute: children[1]);
                break;
            case IR.If @if:
                expr = @if.With(condition: children[^3], then: children[^2], @else: children[^1], paramList: children[..^3].ToArray());
                break;
            default:
                throw new NotSupportedException(enode.Expr.GetType().Name);
        }

        _memo.Add(root, expr);

        return expr;
    }
}

internal sealed class HyperGraph
{
    private readonly Dictionary<int, Dictionary<int, HashSet<ENode>>> _edges;
    private readonly HashSet<int> _nodes;
    private readonly Dictionary<EClass, int> _ids_to_nodes;
    private readonly Dictionary<int, EClass> _nodes_to_ids;
    private int _num_nodes;

    public HyperGraph()
    {
        _edges = new();
        _nodes = new();
        _ids_to_nodes = new();
        _nodes_to_ids = new();
        _num_nodes = 0;
    }

    public bool Contains(EClass id)
    {
        return _ids_to_nodes.ContainsKey(id);
    }

    public List<List<int>> AdjacencyEdges()
    {
        return _edges.Select(kv => kv.Value.Keys.ToList()).ToList();
    }

    public Dictionary<EClass, HashSet<ENode>>? Edges(EClass eclass)
    {
        if (Contains(eclass))
        {
            var result = new Dictionary<EClass, HashSet<ENode>>();
            foreach (var (to, enodes) in _edges[_ids_to_nodes[eclass]])
            {
                result.Add(_nodes_to_ids[to], enodes);
            }

            return result;
        }

        return null;
    }

    public int NumEdges()
    {
        return _edges.Select(kv => kv.Value.Keys.Count).Sum();
    }

    public HashSet<EClass> Nodes()
    {
        return new(_nodes.Select(x => _nodes_to_ids[x]));
    }

    public void AddNode(EClass k)
    {
        var node_id = _num_nodes;
        _ids_to_nodes.Add(k, node_id);
        _nodes_to_ids.Add(node_id, k);
        _edges.Add(node_id, new());
        _nodes.Add(node_id);
        _num_nodes += 1;
    }

    public void Connect(EClass cfrom, EClass cto, ENode enode)
    {
        if (!Contains(cfrom))
        {
            AddNode(cfrom);
        }

        if (!Contains(cto))
        {
            AddNode(cto);
        }

        var from = _ids_to_nodes[cfrom];
        var to = _ids_to_nodes[cto];
        if (!_edges[from].ContainsKey(to))
        {
            _edges[from].Add(to, new HashSet<ENode>(new[] { enode }));
        }
        else
        {
            _edges[from][to].Add(enode);
        }
    }

    public EClass[] Neighbors(EClass u)
    {
        if (Contains(u))
        {
            return _edges[_ids_to_nodes[u]].Keys.Select(x => _nodes_to_ids[x]).ToArray();
        }

        return Array.Empty<EClass>();
    }

    public int GetNodeById(EClass id)
    {
        return _ids_to_nodes[id];
    }

    public EClass GetIdByNode(int node)
    {
        return _nodes_to_ids[node];
    }

    public void RemoveNodeRaw(int node)
    {
        if (_nodes.Contains(node))
        {
            _edges.Remove(node);
            foreach (var (_, v) in _edges)
            {
                v.Remove(node);
            }

            _nodes.Remove(node);
        }
    }

    public void RemoveNode(EClass node)
    {
        var node_id = _ids_to_nodes[node];
        if (Contains(node))
        {
            _edges.Remove(node_id);
            foreach (var (_, v) in _edges)
            {
                v.Remove(node_id);
            }

            _nodes.Remove(node_id);
        }
    }

    public int Size()
    {
        return _nodes.Count;
    }

    public HyperGraph SubGraph(IEnumerable<EClass> nodes)
    {
        var graph = new HyperGraph();
        var node_set = new HashSet<EClass>(nodes);
        foreach (var n in node_set)
        {
            var nedges = _edges[_ids_to_nodes[n]];
            foreach (var (neighbor, enodes) in nedges)
            {
                if (!node_set.Contains(_nodes_to_ids[neighbor]))
                {
                    continue;
                }

                foreach (var enode in enodes)
                {
                    graph.Connect(n, _nodes_to_ids[neighbor], enode);
                }
            }
        }

        return graph;
    }

    public void Dump(string name)
    {
        using (var dumpStream = DumpScope.Current.OpenFile($"Costs/{name}.dot"))
        {
            using var writer = new StreamWriter(dumpStream);
            writer.WriteLine("digraph G {");
            writer.WriteLine("/*");
            writer.WriteLine("[");
            foreach (var (_, v) in _edges.OrderBy(kv => kv.Key))
            {
                writer.WriteLine($"[{string.Join(",", v.Keys)}],");
            }

            writer.WriteLine("]");
            writer.WriteLine("*/");

            foreach (var node in _nodes)
            {
                writer.WriteLine($"{node} [label = \"{_nodes_to_ids[node].Id}\"]");
            }

            foreach (var (u, v) in _edges)
            {
                foreach (var w in v.Keys)
                {
                    writer.WriteLine($"{u} -> {w};");
                }
            }

            writer.WriteLine("}");
        }
    }
}
