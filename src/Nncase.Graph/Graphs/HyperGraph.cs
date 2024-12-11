// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Nncase.Diagnostics;

namespace Nncase.Graphs;

public sealed class HyperGraph<TEClass, TENode>
    where TENode : class
    where TEClass : class
{
    private readonly Dictionary<int, Dictionary<int, HashSet<TENode>>> _edges;
    private readonly HashSet<int> _nodes;
    private readonly Dictionary<TEClass, int> _ids_to_nodes;
    private readonly Dictionary<int, TEClass> _nodes_to_ids;
    private int _num_nodes;

    public HyperGraph()
    {
        _edges = new();
        _nodes = new();
        _ids_to_nodes = new();
        _nodes_to_ids = new();
        _num_nodes = 0;
    }

    public bool Contains(TEClass id)
    {
        return _ids_to_nodes.ContainsKey(id);
    }

    public List<List<int>> AdjacencyEdges()
    {
        return _edges.Select(kv => kv.Value.Keys.ToList()).ToList();
    }

    public Dictionary<TEClass, HashSet<TENode>>? Edges(TEClass eclass)
    {
        if (Contains(eclass))
        {
            var result = new Dictionary<TEClass, HashSet<TENode>>();
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

    public HashSet<TEClass> Nodes()
    {
        return new(_nodes.Select(x => _nodes_to_ids[x]));
    }

    public void AddNode(TEClass k)
    {
        var node_id = _num_nodes;
        _ids_to_nodes.Add(k, node_id);
        _nodes_to_ids.Add(node_id, k);
        _edges.Add(node_id, new());
        _nodes.Add(node_id);
        _num_nodes += 1;
    }

    public void Connect(TEClass cfrom, TEClass cto, TENode enode)
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
            _edges[from].Add(to, new HashSet<TENode>(new[] { enode }));
        }
        else
        {
            _edges[from][to].Add(enode);
        }
    }

    public TEClass[] Neighbors(TEClass u)
    {
        if (Contains(u))
        {
            return _edges[_ids_to_nodes[u]].Keys.Select(x => _nodes_to_ids[x]).ToArray();
        }

        return Array.Empty<TEClass>();
    }

    public int GetNodeById(TEClass id)
    {
        return _ids_to_nodes[id];
    }

    public TEClass GetIdByNode(int node)
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

    public void RemoveNode(TEClass node)
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

    public HyperGraph<TEClass, TENode> SubGraph(IEnumerable<TEClass> nodes)
    {
        var graph = new HyperGraph<TEClass, TENode>();
        var node_set = new HashSet<TEClass>(nodes);
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
                writer.WriteLine($"{node} [label = \"{_nodes_to_ids[node]}\"]");
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

    public List<List<TEClass>> FindCycles()
    {
        var edges = AdjacencyEdges();
        var circuits = new List<List<TEClass>>();
        foreach (var n in Nodes())
        {
            if (Neighbors(n).Contains(n))
            {
                circuits.Add(new() { n });
            }
        }

        var stack = new List<int>();
        bool[] blocked = new bool[NumEdges()];
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
            circuits.Add(stack.Select(GetIdByNode).ToList());
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
