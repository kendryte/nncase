// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections;
using System.Reactive;
using Google.OrTools.ConstraintSolver;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Affine;
using VisitorPatternGenerator;
using Isl = IntegerSetLibrary;

namespace Nncase.Schedule;

public partial interface ITreeNode
{
    public ITreeNode? Parent { get; set; }
}

[Visitor<ITreeNode>]
public partial interface ITreeNodeVisitor<in TArg1, out TReturn>
{
}

public interface ITileAbleNode : ITreeNode
{
    int Level { get; }

    int OpId { get; }

    /// <summary>
    /// Gets the domain var names.
    /// </summary>
    string[] DomainNames { get; }

    /// <summary>
    /// Gets or sets the domain relation which from parent domain map to current node's domain.
    /// </summary>
    Isl.basic_map DomainRelation { get; set; }
}

[Acceptor<ITreeNode, ScopeNode>]
public partial class ScopeNode
{
    private readonly List<ITreeNode> _children;

    public ScopeNode(ITreeNode? parent = null)
    {
        Parent = parent;
        _children = new();
    }

    public ITreeNode? Parent { get; set; }

    public IList<ITreeNode> Children => _children;

    public void Add(ITreeNode node)
    {
        node.Parent = this;
        _children.Add(node);
    }

    public void Insert(int index, ITreeNode node)
    {
        node.Parent = this;
        _children.Insert(index, node);
    }

    public void InsertRange(int index, IList<ITreeNode> nodes)
    {
        foreach (var item in nodes)
        {
            item.Parent = this;
        }

        _children.InsertRange(index, nodes);
    }

    public void Remove(ITreeNode node)
    {
        _children.Remove(node);
        node.Parent = null;
    }
}

[Acceptor<ITreeNode, TileNode>]
public partial class TileNode : ITileAbleNode
{
    private ITreeNode _child;

    public TileNode(int level, int opId, string[] vars)
    {
        Level = level;
        OpId = opId;
        DomainNames = vars;
        DomainRelation = TilingUtilities.GetIdentityMap(vars.Length, $"op{OpId}", $"op{OpId}");
        _child = null!;
    }

    public ITreeNode? Parent { get; set; }

    public int Level { get; }

    public int OpId { get; }

    /// <summary>
    /// Gets the domain var names.
    /// </summary>
    public string[] DomainNames { get; }

    /// <summary>
    /// Gets or sets the domain relation which from parent domain map to current node's domain.
    /// </summary>
    public Isl.basic_map DomainRelation { get; set; }

    public ITreeNode Child
    {
        get => _child; set
        {
            _child = value;
            _child.Parent = this;
        }
    }

    public override string ToString()
    {
        return $"Tile{OpId} @ {Level}";
    }
}

[Acceptor<ITreeNode, OpNode>]
public partial class OpNode : ITileAbleNode
{
    public OpNode(int opId, string[] domainNames, int[] domain, int[][] bufferShapes, Isl.basic_map[] reads, Isl.basic_map write, AffineMap[] accessMaps, Dependence[] dependences)
    {
        Level = 0;
        OpId = opId;
        DomainNames = domainNames;
        DomainRelation = TilingUtilities.GetIdentityMap(domainNames.Length, $"op{OpId}", $"op{OpId}");
        DomainBounds = domain;
        BufferShapes = bufferShapes;
        Reads = reads;
        Write = write;
        Dependences = dependences;
        AccessMaps = accessMaps;
    }

    public ITreeNode? Parent { get; set; }

    public int Level { get; }

    public int OpId { get; }

    /// <summary>
    /// Gets the domain var names.
    /// </summary>
    public string[] DomainNames { get; }

    /// <summary>
    /// Gets or sets the domain relation which from parent domain map to current node's domain.
    /// </summary>
    public Isl.basic_map DomainRelation { get; set; }

    public IReadOnlyList<Dependence> Dependences { get; }

    public IReadOnlyList<int> DomainBounds { get; }

    public int[][] BufferShapes { get; }

    public IReadOnlyList<Isl.basic_map> Reads { get; }

    public Isl.basic_map Write { get; }

    public IReadOnlyList<AffineMap> AccessMaps { get; }

    public override string ToString()
    {
        return $"Op{OpId}";
    }

    /// <summary>
    /// index is current read buffer index.
    /// </summary>
    public record Dependence(int Index, OpNode Node)
    {
    }
}

public static class TreeSearch
{
    public static OpNode BuildTree(Grid current, ScopeNode scope, int level, ref int opId)
    {
        var dependences = new List<OpNode.Dependence>();
        for (int i = 0; i < current.Reads.Length; i++)
        {
            if (current.Reads[i] is Grid producer)
            {
                var producerNode = BuildTree(producer, scope, level, ref opId);
                dependences.Add(new OpNode.Dependence(i, producerNode));
                opId++;
            }
        }

        var bufferShapes = current.Buffers.AsValueEnumerable().Select(TilingUtilities.GetBufferShape).ToArray();
        var domain = TilingUtilities.InferDomainBounds(bufferShapes, current.AccessMaps.ToArray());
        var copId = opId;
        var domainDims = current.AccessMaps[0].Domains.Length;
        var vars = Enumerable.Range(0, domainDims).Select(i => $"op{copId}_d{i}").ToArray();
        var readAccess = new Isl.basic_map[current.AccessMaps.Length - 1];
        for (int i = 0; i < readAccess.Length; i++)
        {
            readAccess[i] = current.AccessMaps[i].AsIslMap($"op{copId}", domain);
        }

        var opNode = new OpNode(copId, vars, domain, bufferShapes, readAccess, current.AccessMaps[^1].AsIslMap($"op{copId}", domain), current.AccessMaps.ToArray(), dependences.ToArray());
        var tileNodeRoot = new TileNode(level, copId, vars);
        TileNode tileNodeTail = tileNodeRoot;
        for (int l = level - 1; l >= 1; l--)
        {
            var child = new TileNode(l, copId, vars);
            tileNodeTail.Child = child;
            tileNodeTail = child;
        }

        tileNodeTail.Child = opNode;
        scope.Add(tileNodeRoot);
        return opNode;
    }

    public static void Dump(ITreeNode tree, string name)
    {
        using (var stream = Diagnostics.DumpScope.Current.OpenFile($"{name}.py"))
        {
            using var writer = new StreamWriter(stream);
            var printer = new TileTreePrinter(writer);
            tree.Accept(printer, TileTreePrinterContext.Default);
            writer.Flush();
        }
    }

    public static void Merge(ITreeNode tree, int opConsumer, int opProducer, int level)
    {
        var merger = new TileTreeMerger(opConsumer, opProducer, level);
        tree.Accept(merger, default);
    }

    public static void Solve(ITreeNode tree, int totalLevel)
    {
        int[] memoryCapacitys = new[] { 2 * 1024 * 1024, int.MaxValue };
        int[] memoryBandWidths = new[] { 256, 128, 4, 1 }; // l0, l1, l2, dram
        var solver = new Solver("treeSolver");
        var one = solver.MakeIntConst(1);
        var zero = solver.MakeIntConst(0);
        var elem = solver.MakeIntConst(4);
        var opNodeMemo = new Dictionary<OpNode, (IntExpr[][] Shapes, IntExpr[] Size)>();
        var tileNodeMemo = new Dictionary<TileNode, TileNodeInfo>();
        var tileableNodeMemo = new Dictionary<ITileAbleNode, DomainInfo>();
        var init = new TileTreeSolverInit(solver, one, zero, elem, opNodeMemo, tileNodeMemo, tileableNodeMemo);
        tree.Accept(init, TileTreeSolverInit.Context.Default);
        var initWrites = new TileTreeSolverInitWrites(solver, one, zero, elem, opNodeMemo, tileNodeMemo, tileableNodeMemo);
        tree.Accept(initWrites, new());

        // 1. each buffer must store one at lowest level.
        // note if parent node reuse buffer C and D, we need to sum the buffer nums of C and D at child node respectively.
        var lowestStoreBufferNumsConstrains = new Dictionary<BufferIdenitity, Constraint>();
        {
            var lowestStoreNums = new Dictionary<BufferIdenitity, IntExpr>();
            foreach (var (tileNode, bufferInfoMemo) in tileNodeMemo)
            {
                foreach (var (bid, bufferInfo) in bufferInfoMemo.BufferInfoMap)
                {
                    if (!lowestStoreNums.TryGetValue(bid, out var nums))
                    {
                        lowestStoreNums.Add(bid, solver.MakeSum(bufferInfo.Place.Select(p => p[0]).ToArray()));
                    }
                    else
                    {
                        lowestStoreNums[bid] += solver.MakeSum(bufferInfo.Place.Select(p => p[0]).ToArray());
                    }
                }
            }

            foreach (var (bid, nums) in lowestStoreNums)
            {
                lowestStoreBufferNumsConstrains[bid] = solver.MakeEquality(nums, 1);
                solver.Add(lowestStoreBufferNumsConstrains[bid]);
            }
        }

        // 2. each tensor only can create one or zero buffer at each create level.
        var eachNodeCreateBufferConstraints = new Dictionary<TileNode, Dictionary<BufferIdenitity, Constraint>>();
        var eachNodeCreateBufferNums = new Dictionary<TileNode, Dictionary<BufferIdenitity, IntExpr>>();
        {
            foreach (var (node, nodeInfo) in tileNodeMemo)
            {
                var createBufferConstraints = eachNodeCreateBufferConstraints[node] = new Dictionary<BufferIdenitity, Constraint>();
                var createBufferNums = eachNodeCreateBufferNums[node] = new Dictionary<BufferIdenitity, IntExpr>();
                foreach (var (bid, bufferInfo) in nodeInfo.BufferInfoMap)
                {
                    if (nodeInfo.DefUseMap.ContainsKey(bid))
                    {
                        // avoid add constraint twice on same buffer.
                        continue;
                    }

                    createBufferNums[bid] = solver.MakeSum(bufferInfo.Place.SelectMany(i => i).ToArray());
                    createBufferConstraints[bid] = solver.MakeLessOrEqual(createBufferNums[bid], 1);
                    createBufferConstraints[bid].SetName($"createCons[{node.Level}, {node.OpId}, {bid}]");
                    solver.Add(createBufferConstraints[bid]);
                }
            }
        }

        // 3. if current level has create a buffer, it's requires previous level store a buffer.
        for (int l = 1; l <= totalLevel; l++)
        {
            foreach (var (childNode, childNodeInfo) in tileNodeMemo.Where(p => p.Key.Level == l))
            {
                foreach (var (childBid, childBufferInfo) in childNodeInfo.BufferInfoMap)
                {
                    if (childNodeInfo.DefUseMap.ContainsKey(childBid))
                    {
                        continue;
                    }

                    foreach (var (parentNode, parentBufferInfo) in tileNodeMemo.Where(p => p.Key.Level == l + 1 && p.Value.BufferInfoMap.ContainsKey(childBid)).Select(p => (p.Key, p.Value.BufferInfoMap[childBid])))
                    {
                        var parentStored = solver.MakeIsEqualVar(solver.MakeSum(parentBufferInfo.Place.Select(p => p[l]).ToArray()), one);
                        var childCreateNums = eachNodeCreateBufferNums[childNode][childBid];
                        var constraint = solver.MakeGreaterOrEqual(childCreateNums, parentStored);
                        constraint.SetName($"dep[{childNode}, {childBid}, {parentNode}]");
                        solver.Add(constraint);
                    }
                }
            }
        }

        // 4. tile var constraints
        foreach (var opNode in opNodeMemo.Keys)
        {
            var tileVarBucket = new Dictionary<string, List<IntVar>>();

            ITileAbleNode? node = opNode;
            while (node is not null)
            {
                var domainInfo = tileableNodeMemo[node];
                for (int i = 0; i < domainInfo.DomainNames.Length; i++)
                {
                    var dimName = domainInfo.DomainNames[i];
                    if (!tileVarBucket.TryGetValue(dimName, out var bucket))
                    {
                        bucket = new();
                        tileVarBucket.Add(dimName, bucket);
                    }

                    bucket.Add(domainInfo.TileVars[i]);
                }

                node = node.GetParentTileableNode();
            }

            for (int i = 0; i < tileableNodeMemo[opNode].DomainNames.Length; i++)
            {
                var dimName = tileableNodeMemo[opNode].DomainNames[i];
                var constraint = solver.MakeEquality(solver.MakeProd(tileVarBucket[dimName].ToArray()), opNode.DomainBounds[i]);
                constraint.SetName($"bound[op{opNode.OpId}, {dimName}]");
                solver.Add(constraint);
            }
        }

        // 5. add the memory capacity constraints
        var memoryCapacityConstraints = new Dictionary<int, Dictionary<TileNode, Constraint>>();
        var memoryCapacitySizes = new Dictionary<int, Dictionary<TileNode, IntExpr>>();
        for (int l = 1; l <= totalLevel; l++)
        {
            memoryCapacityConstraints[l] = new Dictionary<TileNode, Constraint>();
            memoryCapacitySizes[l] = new Dictionary<TileNode, IntExpr>();
            foreach (var (childNode, childNodeInfo) in tileNodeMemo.Where(p => p.Key.Level == l))
            {
                var storedBufferSize = new List<IntExpr>();
                foreach (var (childBid, childBufferInfo) in childNodeInfo.BufferInfoMap)
                {
                    if (childNodeInfo.DefUseMap.ContainsKey(childBid))
                    {
                        continue;
                    }

                    storedBufferSize.AddRange(childBufferInfo.Place.Select(p => p[l - 1]).Zip(childBufferInfo.Size).Select(p => p.First * p.Second));

                    for (int pl = l + 1; pl <= totalLevel; pl++)
                    {
                        foreach (var (parentNode, parentBufferInfo) in tileNodeMemo.Where(p => p.Key.Level == pl && p.Value.BufferInfoMap.ContainsKey(childBid)).Select(p => (p.Key, p.Value.BufferInfoMap[childBid])))
                        {
                            storedBufferSize.AddRange(parentBufferInfo.Place.Select(p => p[l - 1]).Zip(parentBufferInfo.Size).Select(p => p.First * p.Second));
                        }
                    }
                }

                memoryCapacitySizes[l][childNode] = storedBufferSize.Skip(1).Aggregate(storedBufferSize.First(), solver.MakeSum);
                memoryCapacityConstraints[l][childNode] = solver.MakeLessOrEqual(memoryCapacitySizes[l][childNode], memoryCapacitys[l - 1]);
                solver.Add(memoryCapacityConstraints[l][childNode]);
            }
        }

        // compute the cycles as objective
        var levelDataReads = Enumerable.Range(0, totalLevel + 1).Select(i => (IntExpr)zero).ToArray(); // reads[i] mean read from level i+1
        var levelDataWrites = Enumerable.Range(0, totalLevel + 1).Select(i => (IntExpr)zero).ToArray();
        IntExpr computeCycles = zero;
        foreach (var (opNode, opNodeInfo) in opNodeMemo)
        {
            var vars = new List<IntVar>();
            var tileableNode = opNode.GetParentTileableNode();
            while (tileableNode is not null)
            {
                vars.AddRange(tileableNodeMemo[tileableNode].TileVars);
                tileableNode = tileableNode.GetParentTileableNode();
            }

            var loopTrip = solver.MakeProd(vars.ToArray());
            {
                var l0Load = loopTrip * opNodeInfo.Size.SkipLast(1).Aggregate((IntExpr)zero, (acc, s) => acc + s);
                levelDataReads[0] += l0Load; // read from level 1
                var l0Store = loopTrip * opNodeInfo.Size[^1];
                levelDataWrites[0] += l0Store; // write to level 0

                // note use kernel info. amx 32*32 matmul only 4 cycles
                computeCycles += 4 * loopTrip;
            }
        }

        for (int l = 1; l <= totalLevel; l++)
        {
            var writeToCurrent = new List<IntExpr>();
            var readFromPrevious = Enumerable.Range(0, totalLevel - l + 1).Select(i => new List<IntExpr>()).ToArray();
            foreach (var (childNode, childNodeInfo) in tileNodeMemo.Where(p => p.Key.Level == l))
            {
                foreach (var (childBid, childBufferInfo) in childNodeInfo.BufferInfoMap)
                {
                    if (childNodeInfo.DefUseMap.ContainsKey(childBid))
                    {
                        continue;
                    }

                    var writes = childBufferInfo.Place.Select(p => p[l - 1]).Zip(childBufferInfo.Write).Select(p => p.First * p.Second);
                    writeToCurrent.AddRange(writes);
                    readFromPrevious[0].AddRange(writes); // read from level l + 1

                    for (int pl = l + 1; pl <= totalLevel; pl++)
                    {
                        foreach (var (parentNode, parentBufferInfo) in tileNodeMemo.Where(p => p.Key.Level == pl && p.Value.BufferInfoMap.ContainsKey(childBid)).Select(p => (p.Key, p.Value.BufferInfoMap[childBid])))
                        {
                            writes = parentBufferInfo.Place.Select(p => p[l - 1]).Zip(parentBufferInfo.Write).Select(p => p.First * p.Second);
                            writeToCurrent.AddRange(writes);
                            readFromPrevious[pl - l].AddRange(writes); // read from pl + 1
                        }
                    }
                }
            }

            levelDataWrites[l] += writeToCurrent.Skip(1).Aggregate(writeToCurrent.First(), solver.MakeSum);
            for (int i = 0; i < readFromPrevious.Length; i++)
            {
                levelDataReads[l + i] += readFromPrevious[i].Skip(1).Aggregate(readFromPrevious[i].First(), solver.MakeSum);
            }
        }

        var memoryCycles = Enumerable.Range(0, totalLevel + 1).Select(i => (IntExpr)zero).ToArray();
        for (int i = 0; i <= totalLevel; i++)
        {
            memoryCycles[i] += levelDataWrites[i].CeilDiv(memoryBandWidths[i]);
            if (i > 0)
            {
                memoryCycles[i] += levelDataReads[i - 1].CeilDiv(memoryBandWidths[i]);
            }
        }

        var totalCycles = computeCycles;
        for (int l = 0; l <= totalLevel; l++)
        {
            totalCycles = solver.MakeMax(totalCycles, memoryCycles[l]);
        }

        var totalCyclesVar = totalCycles.Var();
        totalCyclesVar.SetRange(1, long.MaxValue / memoryBandWidths[0]); /* avoid crash. */

        var objectiveMonitor = solver.MakeMinimize(totalCyclesVar, 1);
        var logger = solver.MakeSearchLog(1000, totalCyclesVar);
        var collector = solver.MakeNBestValueSolutionCollector(5, false);
        collector.AddObjective(totalCyclesVar);
        var searchAbleVars = new List<IntVar>();
        foreach (var (node, diminfo) in tileableNodeMemo)
        {
            searchAbleVars.AddRange(diminfo.TileVars);
            collector.Add(diminfo.TileVars);
            if (node is TileNode tnode)
            {
                var tnodeInfo = tileNodeMemo[tnode];
                foreach (var (bid, bufferInfo) in tnodeInfo.BufferInfoMap)
                {
                    if (tnodeInfo.DefUseMap.ContainsKey(bid))
                    {
                        continue;
                    }

                    var placeVars = bufferInfo.Place.SelectMany(i => i).ToArray();
                    searchAbleVars.AddRange(placeVars);
                    collector.Add(placeVars);
                    collector.Add(bufferInfo.Size.Select(i => i.Var()).ToArray());
                    collector.Add(bufferInfo.Write.Select(i => i.Var()).ToArray());
                }
            }
        }

        var decisionBuilder = solver.MakeDefaultPhase(searchAbleVars.ToArray());
        var status = solver.Solve(decisionBuilder, new SearchMonitor[] { collector, objectiveMonitor, logger, solver.MakeSolutionsLimit(10) });
        if (status)
        {
            var sol = collector.Solution(collector.SolutionCount() - 1);
            var tileNodeResult = new Dictionary<TileNode, TileNodeAssignment>();
            foreach (var (node, info) in tileNodeMemo)
            {
                var bufferResultMap = new Dictionary<BufferIdenitity, TileNodeBufferAssignment>();
                foreach (var (bid, binfo) in info.BufferInfoMap)
                {
                    var place = new bool[binfo.Place.Length][];
                    for (int i = 0; i < binfo.Place.Length; i++)
                    {
                        place[i] = new bool[binfo.Place[i].Length];
                        for (int j = 0; j < place[i].Length; j++)
                        {
                            place[i][j] = sol.Value(binfo.Place[i][j]) == 1;
                        }
                    }

                    var size = new long[binfo.Size.Length];
                    for (int i = 0; i < size.Length; i++)
                    {
                        size[i] = sol.Value(binfo.Size[i].Var());
                    }

                    var write = new long[binfo.Write.Length];

                    for (int i = 0; i < write.Length; i++)
                    {
                        write[i] = sol.Value(binfo.Write[i].Var());
                    }

                    bufferResultMap[bid] = new TileNodeBufferAssignment(place, size, write);
                }

                tileNodeResult[node] = new TileNodeAssignment(info.DefUseMap, bufferResultMap);
            }

            var tileableNodeResult = new Dictionary<ITileAbleNode, DomainDimAssignment>();
            foreach (var (node, info) in tileableNodeMemo)
            {
                tileableNodeResult[node] = new DomainDimAssignment(info.DomainNames, info.TileVars.Select(sol.Value).ToArray(), info.BufferMasksMap);
            }

            using (var stream = Diagnostics.DumpScope.Current.OpenFile($"result.py"))
            {
                using var writer = new StreamWriter(stream);
                var printer = new TileTreeResultPrinter(writer, tileableNodeResult, tileNodeResult);
                tree.Accept(printer, TileTreeResultPrinter.Context.Default);
                writer.Flush();
            }
        }
    }

    public static void Search(Grid grid)
    {
        var tree = new ScopeNode();
        var opId = 0;
        var maxLevel = 2;
        BuildTree(grid, tree, maxLevel, ref opId);
        Dump(tree, "build");

        // try merge op2 and op1 at level 1
        Merge(tree, 2, 1, 2);
        Dump(tree, "merge_2_1_2");

        Merge(tree, 2, 1, 1);
        Dump(tree, "merge_2_1_1");

        Merge(tree, 2, 0, 2);
        Dump(tree, "merge_2_0_2");

        // // merge 1 0 1
        // Merge(tree, 1, 0, 1);
        // Dump(tree, "merge_1_0_1");
        Solve(tree, maxLevel);
    }
}
