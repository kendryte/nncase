// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Google.OrTools.ConstraintSolver;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.Schedule.TileTree;
using Nncase.TIR;

namespace Nncase.Schedule;

public static class TreeTiler
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
        var dimNames = Enumerable.Range(0, domainDims).Select(i => $"Op{copId}_d{i}").ToArray();
        if (current.Body[0] is not Call { Target: Op op })
        {
            throw new InvalidOperationException("body is not call");
        }

        var opNode = new OpNode(current, op, copId, dimNames, domain, bufferShapes, dependences.ToArray());
        var tileNodeRoot = new TileNode(level, copId, dimNames);
        TileNode tileNodeTail = tileNodeRoot;
        for (int l = level - 1; l >= 1; l--)
        {
            var child = new TileNode(l, copId, dimNames);
            tileNodeTail.Child = child;
            tileNodeTail = child;
        }

        tileNodeTail.Child = opNode;
        scope.Add(tileNodeRoot);
        return opNode;
    }

    public static void DumpAssgin(ITreeNode tree, TreeSolverPrinter printer, Dictionary<OpNode, Constraint[]> tileVarConstraints, Dictionary<BufferIdentity, Constraint> lowestStoreBufferNumsConstrains, Dictionary<TileNode, Dictionary<BufferIdentity, Constraint>> eachParentNodeCreateBufferConstraints, Dictionary<int, Dictionary<TileNode, IntExpr>> levelMemoryUsage, IntExpr[] levelDataReads, IntExpr[] levelDataWrites, IntExpr[] memoryCycles, IntVar computeCycles)
    {
        using (var stream = Diagnostics.DumpScope.Current.OpenFile($"model.py"))
        {
            using var baseWriter = new StreamWriter(stream);
            using var writer = new System.CodeDom.Compiler.IndentedTextWriter(baseWriter, "  ");
            tree.Accept(printer, writer);
            writer.WriteLine("tileVarConstraints:");
            writer.Indent++;
            foreach (var (opnode, consts) in tileVarConstraints)
            {
                TreeSolverPrinter.WriteIntExprVector(writer, opnode.ToString(), consts, printer.Solution);
            }

            writer.Indent--;

            writer.WriteLine("lowestStoreBufferNumsConstrains:");
            writer.Indent++;
            foreach (var (node, cons) in lowestStoreBufferNumsConstrains)
            {
                TreeSolverPrinter.WriteIntExprVector(writer, node.ToString(), new[] { cons }, printer.Solution);
            }

            writer.Indent--;

            writer.WriteLine("EachParentNodeCreateBufferConstraints:");
            writer.Indent++;
            foreach (var (node, constraints) in eachParentNodeCreateBufferConstraints)
            {
                TreeSolverPrinter.WriteIntExprVector(writer, node.ToString(), constraints.Values.ToArray(), printer.Solution);
            }

            writer.Indent--;

            writer.WriteLine("LevelMemoryUsage:");
            writer.Indent++;
            foreach (var (sl, nodeMemoryUsage) in levelMemoryUsage)
            {
                writer.WriteLine($"Level {sl}:");
                writer.Indent++;
                foreach (var (node, usage) in nodeMemoryUsage)
                {
                    TreeSolverPrinter.WriteIntExpr(writer, node.ToString(), usage, printer.Solution);
                }

                writer.Indent--;
            }

            writer.Indent--;

            TreeSolverPrinter.WriteIntExprVector(writer, "LevelDataReads", levelDataReads, printer.Solution);
            TreeSolverPrinter.WriteIntExprVector(writer, "LevelDataWrites", levelDataWrites, printer.Solution);
            TreeSolverPrinter.WriteIntExprVector(writer, "MemoryCycles", memoryCycles, printer.Solution);
            writer.WriteLine($"computeCycles: {computeCycles.ToSimplifyString()}");
        }
    }

    public static TreeSolverResultConstructor? Solve(ITreeNode tree, ITargetOptions targetOptions)
    {
        int[] memoryCapacities = targetOptions.MemoryCapacities;
        int[] memoryBandWidths = targetOptions.MemoryBandWidths;
        var totalLevel = memoryCapacities.Length - 1;
        var argumentsInfo = TreeSolverInitializer.Init(tree, totalLevel, targetOptions, out var solver, out var opNodeMemo, out var tileNodeMemo, out var tileableNodeMemo);
        var initWrites = new TreeSolverWritesInitializer(solver, opNodeMemo, tileNodeMemo, tileableNodeMemo, targetOptions);
        tree.Accept(initWrites, new());

        // 1. each buffer must store one at lowest level.
        // 1.1 count each node's buffer store nums.
        var lowestStoreBufferNums = new Dictionary<TileNode, Dictionary<BufferIdentity, IntExpr>>();
        foreach (var (tileNode, bufferInfoMemo) in tileNodeMemo)
        {
            var tileStoreNums = new Dictionary<BufferIdentity, IntExpr>();
            foreach (var (bid, bufferInfo) in bufferInfoMemo.BufferInfoMap)
            {
                tileStoreNums.Add(bid, solver.MakeSum(bufferInfo.Places.Select(p => p[0]).ToArray()));
            }

            lowestStoreBufferNums.Add(tileNode, tileStoreNums);
        }

        // 1.2 accumulate the child buffer store nums to parent child buffer in each level.
        for (int cl = 1; cl < totalLevel; cl++)
        {
            foreach (var (childNode, childBufferInfoMemo) in tileNodeMemo.Where(t => t.Key.Level == cl))
            {
                if (childNode.GetParentTileableNode() is TileNode parentNode)
                {
                    var partentBufferInfoMemo = tileNodeMemo[parentNode];
                    foreach (var (cbid, cbufferInfo) in childBufferInfoMemo.BufferInfoMap)
                    {
                        var pbid = partentBufferInfoMemo.GetCacheBid(cbid);
                        lowestStoreBufferNums[parentNode][pbid] += lowestStoreBufferNums[childNode][cbid];
                    }
                }
            }
        }

        // 1.3 create buffer store nums constrains at total level.
        var lowestStoreBufferNumsConstrains = new Dictionary<BufferIdentity, Constraint>();
        foreach (var (node, bufferInfoMemo) in tileNodeMemo.Where(t => t.Key.Level == totalLevel))
        {
            foreach (var (bid, bufferInfo) in bufferInfoMemo.BufferInfoMap)
            {
                lowestStoreBufferNumsConstrains[bid] = solver.MakeEquality(lowestStoreBufferNums[node][bid], 1);
                solver.Add(lowestStoreBufferNumsConstrains[bid]);
            }
        }

        // 2. each tensor only can create one or zero buffer at each create level.
        var eachNodeCreateBufferConstraints = new Dictionary<TileNode, Dictionary<BufferIdentity, Constraint>>();
        var eachNodeCreateBufferNums = new Dictionary<TileNode, Dictionary<BufferIdentity, IntExpr>>();
        foreach (var (node, nodeInfo) in tileNodeMemo)
        {
            var createBufferConstraints = eachNodeCreateBufferConstraints[node] = new Dictionary<BufferIdentity, Constraint>();
            var createBufferNums = eachNodeCreateBufferNums[node] = new Dictionary<BufferIdentity, IntExpr>();
            foreach (var (bid, bufferInfo) in nodeInfo.BufferInfoMap)
            {
                createBufferNums[bid] = solver.MakeSum(bufferInfo.Places.SelectMany(i => i).ToArray());
                createBufferConstraints[bid] = solver.MakeLessOrEqual(createBufferNums[bid], 1);
                createBufferConstraints[bid].SetName($"nodeCreate[{node.Level}, {node.OpId}, {bid}]");
                solver.Add(createBufferConstraints[bid]);
            }
        }

        // 2.1 each cache buffer requires it's parent level create a buffer.
        var eachParentNodeCreateBufferConstraints = new Dictionary<TileNode, Dictionary<BufferIdentity, Constraint>>();
        foreach (var (node, nodeInfo) in tileNodeMemo.Where(kv => kv.Key.Level > 1 && kv.Value.DefUseMap.Any()))
        {
            var nodeCreateBufferConstraints = eachParentNodeCreateBufferConstraints[node] = new();
            foreach (var (_, sinkId) in nodeInfo.DefUseMap)
            {
                var cons = nodeCreateBufferConstraints[sinkId] = solver.MakeEquality(solver.MakeSum(nodeInfo.BufferInfoMap[sinkId].Places.Select(p => p[node.Level - 2]).ToArray()), 1);
                cons.SetName($"parentCreate[{node.Level}, {node.OpId}, {sinkId}]");
                solver.Add(cons);
            }
        }

        // 3. if current level has create a buffer, it's requires parent level store a buffer in current level.
        for (int l = 1; l < totalLevel; l++)
        {
            foreach (var (childNode, childNodeInfo) in tileNodeMemo.Where(p => p.Key.Level == l))
            {
                if (childNode.GetParentTileableNode() is TileNode parentNode && parentNode.Level == l + 1)
                {
                    // because of we assume the inputs/outputs buffer already stores at total Level.
                    if (parentNode.Level == totalLevel)
                    {
                        continue;
                    }

                    foreach (var (childBid, childBufferInfo) in childNodeInfo.BufferInfoMap)
                    {
                        var parentNodeInfo = tileNodeMemo[parentNode];
                        var pbid = parentNodeInfo.GetCacheBid(childBid);
                        var parentStored = solver.MakeIsEqualVar(solver.MakeSum(parentNodeInfo.BufferInfoMap[pbid].Places.Select(p => p[l - 1]).ToArray()), solver.MakeIntConst(1));
                        var childCreateNums = eachNodeCreateBufferNums[childNode][childBid];
                        var constraint = solver.MakeEquality(childCreateNums, parentStored);
                        constraint.SetName($"dep[{childNode}, {childBid}, {parentNode}]");
                        solver.Add(constraint);
                    }
                }
            }
        }

        // 4. tile var constraints
        var tileVarConstraints = new Dictionary<OpNode, Constraint[]>();
        foreach (var opNode in opNodeMemo.Keys)
        {
            var domainInfo = tileableNodeMemo[opNode];
            var constraints = new Constraint[domainInfo.TileVars.Length];
            for (int i = 0; i < domainInfo.TileVars.Length; i++)
            {
                var dimName = opNode.DimNames[i];
                constraints[i] = solver.MakeEquality(domainInfo.ForwardExtents[i], opNode.DomainBounds[i]);
                constraints[i].SetName($"bound[op{opNode.OpId}, {dimName}]");
                solver.Add(constraints[i]);
            }

            tileVarConstraints.Add(opNode, constraints);
        }

        // 5. add the memory schedule constraints
        var levelTreeBufferSizes = new Dictionary<int, Dictionary<TileNode, Dictionary<(TileNode Node, BufferIdentity Buffer), IntExpr>>>();
        var levelTreeBufferLifeness = new Dictionary<int, Dictionary<TileNode, Dictionary<(TileNode Node, BufferIdentity Buffer), Tuple<int, int>>>>();
        var levelMemoryUsage = new Dictionary<int, Dictionary<TileNode, IntExpr>>();
        for (int sl = 1; sl < totalLevel; sl++)
        {
            var treeBufferSizes = levelTreeBufferSizes[sl] = new();
            var treeBufferLifeness = levelTreeBufferLifeness[sl] = new();
            var nodeMemoryUsage = levelMemoryUsage[sl] = new();

            // We collect the buffer schedule information of each memory level from each root tile node to the child node.
            foreach (var (rootNode, rootNodeInfo) in tileNodeMemo.Where(p => p.Key.Level == totalLevel))
            {
                var nodeBufferSizes = treeBufferSizes[rootNode] = new();
                var nodeBufferLiveness = treeBufferLifeness[rootNode] = new();
                var beginTime = int.MaxValue;
                var endTime = int.MinValue;

                foreach (var (bid, bufferInfo) in rootNodeInfo.BufferInfoMap)
                {
                    var extents = bufferInfo.Places.Select(p => p[sl - 1]).Zip(bufferInfo.SizeVars).Select(p => p.First * p.Second).ToArray();
                    nodeBufferSizes[(rootNode, bid)] = extents.Skip(1).Aggregate(extents[0], solver.MakeSum);
                    nodeBufferLiveness[(rootNode, bid)] = bufferInfo.Liveness;
                    beginTime = Math.Min(beginTime, bufferInfo.Liveness.Item1);
                    endTime = Math.Max(endTime, bufferInfo.Liveness.Item2);
                }

                rootNode.Child.Walk(current =>
                {
                    if (current is not TileNode { Level: >= 1 } childNode)
                    {
                        return;
                    }

                    foreach (var (cbid, childBufferInfo) in tileNodeMemo[childNode].BufferInfoMap)
                    {
                        // accumulate the extents
                        var extents = childBufferInfo.Places.Select(p => p[sl - 1]).Zip(childBufferInfo.SizeVars).Select(p => p.First * p.Second).ToArray();
                        nodeBufferSizes[(childNode, cbid)] = extents.Skip(1).Aggregate(extents[0], solver.MakeSum);
                        nodeBufferLiveness[(childNode, cbid)] = childBufferInfo.Liveness;
                        beginTime = Math.Min(beginTime, childBufferInfo.Liveness.Item1);
                        endTime = Math.Max(endTime, childBufferInfo.Liveness.Item2);
                    }
                });

                // Add constraints according to liveness.
#if false
                DumpGantt(nodeBufferSizes, nodeBufferLiveness, rootNode, sl);
#endif

                var lastTimeStamp = new HashSet<(TileNode Node, BufferIdentity Buffer)>();
                for (int i = beginTime; i <= endTime; i++)
                {
                    var curTimeStamp = new HashSet<(TileNode Node, BufferIdentity Buffer)>();
                    foreach (var (key, liveness) in nodeBufferLiveness)
                    {
                        if (i >= liveness.Item1 && i <= liveness.Item2)
                        {
                            curTimeStamp.Add(key);
                        }
                    }

                    if (!lastTimeStamp.SetEquals(curTimeStamp))
                    {
                        var bufs = curTimeStamp.Select(key => nodeBufferSizes[key]).ToArray();
                        var size = bufs.Skip(1).Aggregate(bufs.First(), solver.MakeSum);
                        var cons = solver.MakeLessOrEqual(size, memoryCapacities[sl]);
                        solver.Add(cons);
                        lastTimeStamp.Clear(); // update last stamp.
                        lastTimeStamp.UnionWith(curTimeStamp);
                    }
                }
            }
        }

        // compute the cycles as objective
        var levelDataReads = Enumerable.Range(0, totalLevel + 1).Select(i => (IntExpr)solver.MakeIntConst(0)).ToArray();
        var levelDataWrites = Enumerable.Range(0, totalLevel + 1).Select(i => (IntExpr)solver.MakeIntConst(0)).ToArray();
        IntExpr computeCycles = solver.MakeIntConst(0);

        // the l0 write and l1 read update by op node. l0 have no reads.
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
                var l1Load = loopTrip * opNodeInfo.Size.SkipLast(1).Aggregate((IntExpr)solver.MakeIntConst(0), (acc, s) => acc + s);
                levelDataReads[1] += l1Load; // read from level 1
                var l0Store = loopTrip * opNodeInfo.Size[^1];
                levelDataWrites[0] += l0Store; // write to level 0

                // note use kernel info. amx 32*32 matmul only 4 cycles
                computeCycles += 4 * loopTrip;
            }
        }

        // from top to down.
        foreach (var (topNode, _) in tileNodeMemo.Where(p => p.Key.Level == totalLevel))
        {
            void UpdateLevelReadWrites(ITreeNode treeNode)
            {
                if (treeNode is not TileNode tileNode)
                {
                    return;
                }

                var currentLevel = tileNode.Level;
                var nodeWrites = Enumerable.Range(0, totalLevel + 1).Select(_ => new List<IntExpr>()).ToArray();
                var nodeReads = Enumerable.Range(0, totalLevel + 1).Select(_ => new List<IntExpr>()).ToArray();
                var nodeInfo = tileNodeMemo[tileNode];
                foreach (var (bid, bufferInfo) in nodeInfo.BufferInfoMap)
                {
                    for (int sl = 0; sl < currentLevel; sl++)
                    {
                        if (sl >= bufferInfo.Places[0].Length)
                        {
                            continue;
                        }

                        var loopsWrites = bufferInfo.Places.Select(p => p[sl]).Zip(bufferInfo.Writes).Select(p => p.First * p.Second).ToArray();
                        var write = loopsWrites.Skip(1).Aggregate(loopsWrites[0], solver.MakeSum);
                        nodeWrites[sl + 1].Add(write); // write at store level (sl + 1).
                        if (currentLevel + 1 <= totalLevel)
                        {
                            // can't read from top level's outside.
                            nodeReads[currentLevel + 1].Add(write); // read from current level + 1.
                        }
                    }
                }

                for (int l = 0; l < totalLevel + 1; l++)
                {
                    if (nodeWrites[l].Any())
                    {
                        levelDataWrites[l] += nodeWrites[l].Skip(1).Aggregate(nodeWrites[l].First(), solver.MakeSum);
                    }

                    if (nodeReads[l].Any())
                    {
                        levelDataReads[l] += nodeReads[l].Skip(1).Aggregate(nodeReads[l].First(), solver.MakeSum);
                    }
                }
            }

            topNode.Walk(UpdateLevelReadWrites, false);
        }

        var memoryCycles = Enumerable.Range(0, totalLevel + 1).Select(i => (IntExpr)solver.MakeIntConst(0)).ToArray();
        for (int i = 0; i <= totalLevel; i++)
        {
            if (i < totalLevel)
            {
                // haven't write to totalLevel
                memoryCycles[i] += levelDataWrites[i].CeilDiv(memoryBandWidths[i]);
            }

            if (i > 0)
            {
                // haven't read from l0
                memoryCycles[i] += levelDataReads[i].CeilDiv(memoryBandWidths[i]);
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
        var collector = solver.MakeNBestValueSolutionCollector(5, false);
        collector.AddObjective(totalCyclesVar);
        collector.Add(levelDataReads.Select(i => i.Var()).ToArray());
        collector.Add(levelDataWrites.Select(i => i.Var()).ToArray());
        collector.Add(memoryCycles.Select(i => i.Var()).ToArray());
        collector.Add(computeCycles.Var());
        var searchAbleVars = new List<IntVar>();
        foreach (var (node, diminfo) in tileableNodeMemo)
        {
            searchAbleVars.AddRange(diminfo.TileVars);
            collector.Add(diminfo.TileVars);
            collector.Add(diminfo.ForwardExtents.Select(x => x.Var()).ToArray());
        }

        foreach (var (node, info) in opNodeMemo)
        {
            collector.Add(info.Shapes.SelectMany(i => i).Select(i => i.Var()).ToArray());
            collector.Add(info.Size.Select(i => i.Var()).ToArray());
        }

        foreach (var (node, info) in tileNodeMemo)
        {
            foreach (var (bid, bufferInfo) in info.BufferInfoMap)
            {
                var placeVars = bufferInfo.Places.SelectMany(i => i).ToArray();
                searchAbleVars.AddRange(placeVars);
                collector.Add(placeVars);
                collector.Add(bufferInfo.Shapes.SelectMany(i => i).Select(i => i.Var()).ToArray());
                collector.Add(bufferInfo.Writes.Select(i => i.Var()).ToArray());
                collector.Add(bufferInfo.SizeVars.Select(i => i.Var()).ToArray());
                collector.Add(bufferInfo.SizeExprs.Select(i => i.Var()).ToArray());
            }
        }

        foreach (var (_, treeBufferSizes) in levelTreeBufferSizes)
        {
            foreach (var (_, nodeBufferSizes) in treeBufferSizes)
            {
                foreach (var (_, bufferSize) in nodeBufferSizes)
                {
                    collector.Add(bufferSize.Var());
                }
            }
        }

        var decisionBuilder = solver.MakeDefaultPhase(searchAbleVars.ToArray());
        var status = solver.Solve(decisionBuilder, new SearchMonitor[] { collector, objectiveMonitor, solver.MakeSolutionsLimit(10), solver.MakeTimeLimit(30000),
#if DEBUG
        solver.MakeSearchLog(10000, totalCyclesVar),
#endif
         });
        if (!status)
        {
            return null;
        }

        var sol = collector.Solution(collector.SolutionCount() - 1);

        // dump model
        // builder IR
#if false
        DumpAssgin(tree, new TreeSolverPrinter(sol, solver, opNodeMemo, tileNodeMemo, tileableNodeMemo, compileOptions.TargetOptions), tileVarConstraints, lowestStoreBufferNumsConstrains, eachParentNodeCreateBufferConstraints, levelMemoryUsage, levelDataReads, levelDataWrites, memoryCycles, totalCyclesVar);
#endif

        return new TreeSolverResultConstructor(tree, sol.ObjectiveValue(), sol, argumentsInfo, levelTreeBufferSizes, levelTreeBufferLifeness, solver, opNodeMemo, tileNodeMemo, tileableNodeMemo, targetOptions);
    }

    public static void DumpGantt(Dictionary<(TileNode Node, BufferIdentity Buffer), IntExpr> nodeBufferSizes, Dictionary<(TileNode Node, BufferIdentity Buffer), Tuple<int, int>> nodeBufferLiveness, TileNode rootNode, int storeLevel)
    {
        string GetStartStr(string name, int start) => $"[{name}] starts D+{start}";
        string GetDurationStr(string name, int duration) => $"[{name}] requires {duration} days";
        using (var fs = Diagnostics.DumpScope.Current.OpenFile($"Op{rootNode.OpId}_{rootNode.Level}_store_{storeLevel}_gantt.md"))
        {
            using var writer = new StreamWriter(fs);
            writer.WriteLine("```plantuml");
            writer.WriteLine("@startgantt");
            writer.WriteLine("printscale daily zoom 10");

            foreach (var ((node, bid), liveness) in nodeBufferLiveness)
            {
                var name = $"cl{node.Level} op{bid.Node.OpId} {bid.Index}";
                writer.WriteLine(GetDurationStr(name, liveness.Item2 - liveness.Item1));
                writer.WriteLine(GetStartStr(name, liveness.Item1));
            }

            writer.WriteLine("@endgantt");
            writer.WriteLine("```");
        }
    }

    public static List<MergePoint> EnumerateMergePoint(ITreeNode tree, int level)
    {
        var collector = new TreeMergePointCollector(level);
        tree.Accept(collector, default);
        return collector.Points;
    }

    public static Call Tile(Grid grid, string moduleKind, int itemNumber, ITargetOptions targetOptions)
    {
        // var root = new ScopeNode();
        // var opId = 0;
        var totalLevel = targetOptions.MemoryCapacities.Length - 1;
        var root = TreeBuilder.Build(grid, totalLevel);

        // BuildTree(grid, root, totalLevel, ref opId);
        if (Diagnostics.DumpScope.Current.IsEnabled(Diagnostics.DumpFlags.Tiling))
        {
            root.Dump($"device_func{itemNumber}_original");
        }

        TreeSolverResultConstructor? bestConstructor = null;
#if false
        root.Merge(1, 0, 2);
        root.Merge(1, 0, 1);
        bestConstructor = Solve(root, targetOptions);
#else
        foreach (var chunk in EnumerateAll(root, totalLevel, new()).Chunk(System.Math.Max(System.Environment.ProcessorCount - 2, 1)))
        {
            foreach (var resultConstructor in chunk.AsParallel().Select(isoTree => Solve(isoTree.Root, targetOptions)).OfType<TreeSolverResultConstructor>())
            {
                bestConstructor = (bestConstructor?.ObjectiveValue <= resultConstructor.ObjectiveValue ? bestConstructor : resultConstructor) ?? resultConstructor;
            }
        }
#endif

        if (bestConstructor is null)
        {
            throw new InvalidOperationException("can't solver!");
        }

        if (Diagnostics.DumpScope.Current.IsEnabled(Diagnostics.DumpFlags.Tiling))
        {
            bestConstructor.Tree.Dump($"device_func{itemNumber}_best");
        }

        return bestConstructor.ConstructResult(moduleKind, itemNumber);
    }

    private static List<IsomorphicTree> EnumerateAll(ITreeNode tree, int totalLevel, List<MergePoint> path)
    {
        var result = new List<IsomorphicTree>() { new(tree, new(path)) };
        for (int level = totalLevel; level > 0; level--)
        {
            var points = EnumerateMergePoint(tree, level);
            var isoTrees = new List<IsomorphicTree>();
            foreach (var p in points)
            {
                var cloned = tree.Root<ITreeNode>().Clone();
                if (cloned.Merge(p.Consumer, p.Producer, level))
                {
                    // Dump(cloned, p.ToString());
                    isoTrees.Add(new(cloned, new(path) { p }));
                }
            }

            result.AddRange(isoTrees.Select(isoTree => EnumerateAll(isoTree.Root, level, isoTree.Path)).SelectMany(i => i));
        }

        return result;
    }

    private record IsomorphicTree(ITreeNode Root, List<MergePoint> Path)
    {
    }
}
