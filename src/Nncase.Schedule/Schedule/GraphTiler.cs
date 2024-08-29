// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Google.OrTools.ConstraintSolver;
using NetFabric.Hyperlinq;
using Nncase.Graphs;
using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.Schedule.TileGraph;
using Nncase.TIR;
using QuikGraph;
using QuikGraph.Algorithms;
using QuikGraph.Graphviz;

namespace Nncase.Schedule;

public sealed class GraphTiler
{
    private readonly Dictionary<TileNode, PrimFunctionWrapper> _primFuncMemo = new(new ITreeNodeComparer());

    private int _useCached;

    public Expr Tile(Expr preExpr, string moduleKind, int itemNumber, ICpuTargetOptions targetOptions)
    {
        var totalLevel = targetOptions.MemoryCapacities.Length - 1;
        var rootGraph = GraphBuilder.Build(preExpr, totalLevel, out var exprMemo);
        if (Diagnostics.DumpScope.Current.IsEnabled(Diagnostics.DumpFlags.Tiling))
        {
            rootGraph.Dump($"device_func{itemNumber}_original");
        }

        // bufferize root graph.
        var bufferGraphMemo = rootGraph.Bufferize();
        if (Diagnostics.DumpScope.Current.IsEnabled(Diagnostics.DumpFlags.Tiling))
        {
            bufferGraphMemo[rootGraph].Dump($"device_func{itemNumber}_original_buffer");
        }

#if true
        // condense the root graph.
        var condensedGraph = rootGraph.Condense();
        if (Diagnostics.DumpScope.Current.IsEnabled(Diagnostics.DumpFlags.Tiling))
        {
            using (var file = Diagnostics.DumpScope.Current.OpenFile($"device_func{itemNumber}_condensed.dot"))
            {
                using var writer = new StreamWriter(file);
                writer.Write(condensedGraph.ToGraphviz(init =>
                {
                    init.FormatVertex += (_, arg) =>
                    {
                        if (arg.Vertex is TieredTileGraph t)
                        {
                            arg.VertexFormat.Label = t.ToString();
                        }
                    };
                }));
            }
        }

        // convert root graph as tree.
        var rootTree = TileNode.FromTileGraph(rootGraph, out var treeGraphMemo);

        var argumentsMemo = bufferGraphMemo[rootGraph].GetInputsOutputs().Inputs.ToDictionary(k => k, k => k.Node.Grid.GetArgument(k.Index));
        var resultMemo = new Dictionary<TieredTileGraph, Expr>();
        foreach (var (primGraph, i) in condensedGraph.TopologicalSort().Select((s, i) => (s, i)))
        {
            var primTree = treeGraphMemo[primGraph];
            var primBufferGraph = bufferGraphMemo[primGraph];
            var (inputBids, outputBids) = primBufferGraph.GetInputsOutputs();

            if (!_primFuncMemo.TryGetValue(primTree, out var wrapper))
            {
                var result = SolvePrimGraph(primTree, primBufferGraph, targetOptions);
                result.ScheduleBuffers();
                var bodyBuilder = T.Sequential();
                result.Visit(primTree, new(bodyBuilder, Array.Empty<Expr>()));
                var parameters = inputBids.Concat(outputBids).Select(k => result.PrimBufferMemo[k]).ToArray();
                var funcBuilder = T.PrimFunc($"device_func{itemNumber}_{i}", moduleKind, parameters).Body(bodyBuilder);
                var primFunc = funcBuilder.Build();
                wrapper = new PrimFunctionWrapper(primFunc, inputBids.Count, inputBids.Concat(outputBids).Select(bid => bid.Node.Grid.GetArgument(bid.Index).CheckedType).ToArray());
                _primFuncMemo.Add(primTree, wrapper);
            }
            else
            {
                _useCached++;
            }

            var finalCall = new Call(wrapper, inputBids.Select(bid => argumentsMemo[bid]).ToArray());
            resultMemo.Add(primGraph, finalCall);

            // save the output.
            foreach (var outputBid in outputBids)
            {
                if (!argumentsMemo.TryGetValue(outputBid, out var _))
                {
                    foreach (var outEdge in bufferGraphMemo[rootGraph].OutEdges(outputBid).Where(e => e.Tag is BufferEdgeKind.Outer))
                    {
                        argumentsMemo.Add(outEdge.Target, finalCall);
                    }

                    argumentsMemo.Add(outputBid, finalCall);
                }
            }
        }

        var cloner = new ReplacingExprCloner(exprMemo.ToDictionary(kv => (Expr)kv.Key, kv => resultMemo[kv.Value]));
        return cloner.Clone(preExpr, default);

#else
        PrimGraphSolveResult? bestConstructor = null;
        foreach (var chunk in EnumerateAll(root, totalLevel, new()).Chunk(System.Math.Max(System.Environment.ProcessorCount - 2, 1)))
        {
            foreach (var resultConstructor in chunk.AsParallel().Select(isoTree => Solve(isoTree.Root, targetOptions)).OfType<GraphSolverResultConstructor>())
            {
                bestConstructor = (bestConstructor?.ObjectiveValue <= resultConstructor.ObjectiveValue ? bestConstructor : resultConstructor) ?? resultConstructor;
            }
        }
#endif

        // if (bestConstructor is null)
        // {
        //     throw new InvalidOperationException("can't solver!");
        // }

        // if (Diagnostics.DumpScope.Current.IsEnabled(Diagnostics.DumpFlags.Tiling))
        // {
        //     bestConstructor.Tree.Dump($"device_func{itemNumber}_best");
        // }

        // return bestConstructor.ConstructResult(moduleKind, itemNumber);
        // return new Call(None.Default);
    }

    private TreeSolveResult SolvePrimGraph(TileNode primTree, BufferGraph primBufferGraph, ICpuTargetOptions targetOptions)
    {
        int[] memoryCapacities = targetOptions.MemoryCapacities;
        int[] memoryBandWidths = targetOptions.MemoryBandWidths;
        var totalLevel = memoryCapacities.Length - 1;
        TreeSolverInitializer.Init(primTree, totalLevel, targetOptions, out var solver, out var opNodeMemo, out var tileNodeMemo, out var tileableNodeMemo);
        var initWrites = new TreeSolverWritesInitializer(solver, opNodeMemo, tileNodeMemo, tileableNodeMemo, targetOptions);
        initWrites.Visit(primTree, new());

        // 1. each buffer must store one at lowest level.
        // 1.1 count each node's buffer store nums.
        var lowestStoreBufferNums = new Dictionary<TileNode, Dictionary<BufferIdentity, IntExpr>>();
        foreach (var (tileNode, bufferInfoMemo) in tileNodeMemo)
        {
            var tileStoreNums = new Dictionary<BufferIdentity, IntExpr>();
            foreach (var (bid, bufferInfo) in bufferInfoMemo.BufferInfoMap)
            {
                tileStoreNums.Add(bid, solver.MakeSum(bufferInfo.Places.Select(p => p[0].Var()).ToArray()));
            }

            lowestStoreBufferNums.Add(tileNode, tileStoreNums);
        }

        // 1.2 accumulate the child buffer store nums to parent child buffer in each level.
        for (int cl = 1; cl < totalLevel; cl++)
        {
            foreach (var (childNode, childBufferInfoMemo) in tileNodeMemo.Where(t => t.Key.Level == cl))
            {
                if (childNode.Parent is TileNode parentNode && parentNode.OpId != -1)
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
                createBufferNums[bid] = solver.MakeSum(bufferInfo.Places.SelectMany(i => i).Select(i => i.Var()).ToArray());
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
                var cons = nodeCreateBufferConstraints[sinkId] = solver.MakeEquality(solver.MakeSum(nodeInfo.BufferInfoMap[sinkId].Places.Select(p => p[node.Level - 2].Var()).ToArray()), 1);
                cons.SetName($"parentCreate[{node.Level}, {node.OpId}, {sinkId}]");
                solver.Add(cons);
            }
        }

        // 3. if current level has create a buffer, it's requires parent level store a buffer in current level.
        for (int l = 1; l < totalLevel; l++)
        {
            foreach (var (childNode, childNodeInfo) in tileNodeMemo.Where(p => p.Key.Level == l))
            {
                if (childNode.Parent is TileNode parentNode && parentNode.Level == l + 1)
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
                        var parentStored = solver.MakeIsEqualVar(solver.MakeSum(parentNodeInfo.BufferInfoMap[pbid].Places.Select(p => p[l - 1].Var()).ToArray()), solver.MakeIntConst(1));
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
                constraints[i] = solver.MakeEquality(domainInfo.ForwardExtents[i], opNode.DomainBounds[i]);
                constraints[i].SetName($"bound[op{opNode.OpId}, d{i}]");
                solver.Add(constraints[i]);
            }

            tileVarConstraints.Add(opNode, constraints);
        }

        // 5. add the memory schedule constraints, each level has own memory plan schedule.
        var levelBufferSizes = new Dictionary<int, Dictionary<NodeWithBuffer, IntExpr>>();
        var levelBufferLifeness = new Dictionary<int, Dictionary<NodeWithBuffer, Tuple<int, int>>>();
        for (int sl = 1; sl < totalLevel; sl++)
        {
            // note currently there is a only one root
            var nodeBufferSizes = levelBufferSizes[sl] = new();
            var nodeBufferLiveness = levelBufferLifeness[sl] = new();
            var rootNodeInfo = tileNodeMemo[primTree];
            var beginTime = int.MaxValue;
            var endTime = int.MinValue;

            foreach (var (bid, bufferInfo) in rootNodeInfo.BufferInfoMap)
            {
                var extents = bufferInfo.Places.Select(p => p[sl - 1]).Zip(bufferInfo.SizeVars).Select(p => p.First * p.Second).ToArray();
                nodeBufferSizes[new(primTree, bid)] = extents.Skip(1).Aggregate(extents[0], solver.MakeSum);
                nodeBufferLiveness[new(primTree, bid)] = bufferInfo.Liveness;
                beginTime = Math.Min(beginTime, bufferInfo.Liveness.Item1);
                endTime = Math.Max(endTime, bufferInfo.Liveness.Item2);
            }

            primTree.Walk(current =>
            {
                if (ReferenceEquals(current, primTree))
                {
                    return;
                }

                if (current is not TileNode { Level: >= 1 } childNode)
                {
                    return;
                }

                foreach (var (cbid, childBufferInfo) in tileNodeMemo[childNode].BufferInfoMap)
                {
                    // accumulate the extents
                    var extents = childBufferInfo.Places.Select(p => p[sl - 1]).Zip(childBufferInfo.SizeVars).Select(p => p.First * p.Second).ToArray();
                    nodeBufferSizes[new(childNode, cbid)] = extents.Skip(1).Aggregate(extents[0], solver.MakeSum);
                    nodeBufferLiveness[new(childNode, cbid)] = childBufferInfo.Liveness;
                    beginTime = Math.Min(beginTime, childBufferInfo.Liveness.Item1);
                    endTime = Math.Max(endTime, childBufferInfo.Liveness.Item2);
                }
            });

            // Add constraints according to liveness.
#if false
                DumpGantt(nodeBufferSizes, nodeBufferLiveness, primGraph, sl);
#endif

            var lastTimeStamp = new HashSet<NodeWithBuffer>();
            for (int i = beginTime; i <= endTime; i++)
            {
                var curTimeStamp = new HashSet<NodeWithBuffer>();
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

        // compute the cycles as objective
        var levelDataReads = Enumerable.Range(0, totalLevel + 1).Select(i => (IntExpr)solver.MakeIntConst(0)).ToArray();
        var levelDataWrites = Enumerable.Range(0, totalLevel + 1).Select(i => (IntExpr)solver.MakeIntConst(0)).ToArray();
        IntExpr computeCycles = solver.MakeIntConst(0);

        // the l0 write and l1 read update by op node. l0 have no reads.
        foreach (var (opNode, opNodeInfo) in opNodeMemo)
        {
            var vars = new List<IntVar>();
            var tileableNode = opNode.Parent;
            while (tileableNode is TileNode g && g.OpId != -1)
            {
                vars.AddRange(tileableNodeMemo[tileableNode].TileVars.Select(i => i.Var()));
                tileableNode = tileableNode.Parent;
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
            void UpdateLevelReadWrites(ITileable treeNode)
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
            searchAbleVars.AddRange(diminfo.TileVars.Select(i => i.Var()));
            collector.Add(diminfo.TileVars.Select(i => i.Var()).ToArray());
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
                searchAbleVars.AddRange(placeVars.Select(i => i.Var()));
                collector.Add(placeVars.Select(i => i.Var()).ToArray());
                collector.Add(bufferInfo.Shapes.SelectMany(i => i).Select(i => i.Var()).ToArray());
                collector.Add(bufferInfo.Writes.Select(i => i.Var()).ToArray());
                collector.Add(bufferInfo.SizeVars.Select(i => i.Var()).ToArray());
                collector.Add(bufferInfo.SizeExprs.Select(i => i.Var()).ToArray());
            }
        }

        foreach (var (_, nodeBufferSizes) in levelBufferSizes)
        {
            foreach (var (_, bufferSize) in nodeBufferSizes)
            {
                collector.Add(bufferSize.Var());
            }
        }

        var decisionBuilder = solver.MakeDefaultPhase(searchAbleVars.ToArray());
        var monitors = new List<SearchMonitor>() { collector, objectiveMonitor, solver.MakeSolutionsLimit(10), solver.MakeTimeLimit(30000) };
        if (Diagnostics.DumpScope.Current.IsEnabled(Diagnostics.DumpFlags.Tiling))
        {
            monitors.Add(solver.MakeSearchLog(10000, totalCyclesVar));
        }

        var status = solver.Solve(decisionBuilder, monitors.ToArray());
        if (!status)
        {
            // return null;
        }

        var sol = collector.Solution(collector.SolutionCount() - 1);

        // dump model
        // builder IR
#if false
        DumpAssgin(tree, new GraphSolverPrinter(sol, solver, opNodeMemo, tileNodeMemo, tileableNodeMemo, compileOptions.TargetOptions), tileVarConstraints, lowestStoreBufferNumsConstrains, eachParentNodeCreateBufferConstraints, levelMemoryUsage, levelDataReads, levelDataWrites, memoryCycles, totalCyclesVar);
#endif

        var levelBufferSizesAssgin = levelBufferSizes.ToDictionary(kv => kv.Key, kv => kv.Value.ToDictionary(p => p.Key, p => sol.Value(p.Value.Var())));
        var opNodeMemoAssgin = opNodeMemo.ToDictionary(kv => kv.Key, kv => new OpNodeInfo<long>(kv.Value.Maps, sol.Value(kv.Value.Shapes), sol.Value(kv.Value.Size)));
        var tileNodeMemoAssgin = tileNodeMemo.ToDictionary(kv => kv.Key, kv => new TileNodeInfo<long>(sol.Value(kv.Value.BackWardExtents), kv.Value.DefUseMap, kv.Value.BufferInfoMap.ToDictionary(p => p.Key, p => new TileNodeBufferInfo<long>(p.Value.Liveness, p.Value.Map, sol.Value(p.Value.Places), sol.Value(p.Value.Shapes), sol.Value(p.Value.Writes), sol.Value(p.Value.SizeVars), sol.Value(p.Value.SizeExprs), p.Value.Masks))));
        var tileableNodeMemoAssgin = tileableNodeMemo.ToDictionary(kv => kv.Key, kv => new DomainInfo<long>(sol.Value(kv.Value.TileVars), sol.Value(kv.Value.ForwardExtents), kv.Value.DimsMap));

        // ScheduledResultMemo.Add(primGraph, );
        return new TreeSolveResult(primBufferGraph, sol.ObjectiveValue(), levelBufferSizesAssgin, levelBufferLifeness, opNodeMemoAssgin, tileNodeMemoAssgin, tileableNodeMemoAssgin, targetOptions);
    }

    private void DumpGantt(Dictionary<(TieredTileGraph Node, BufferIdentity Buffer), IntExpr> nodeBufferSizes, Dictionary<(TieredTileGraph Node, BufferIdentity Buffer), Tuple<int, int>> nodeBufferLiveness, TieredTileGraph rootNode, int storeLevel)
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
}
