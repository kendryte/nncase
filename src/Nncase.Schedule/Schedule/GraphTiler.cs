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

public class GraphTiler
{
    public int DeviceFuncionCount { get; private set; }

    public Dictionary<TileNode, TiledFunc> SolveMemo { get; } = new Dictionary<TileNode, TiledFunc>(new ITreeNodeComparer());

    public static TreeSolveResult SolvePrimGraph(TileNode primTree, Dictionary<TieredTileGraph, BufferGraph> bufferGraphMemo, ICpuTargetOptions targetOptions, string moduleKind)
    {
        int[] memoryCapacities = targetOptions.MemoryCapacities;
        int[] memoryBandWidths = targetOptions.MemoryBandWidths;
        var topLevel = memoryCapacities.Length;
        TreeSolverInitializer.Init(primTree, bufferGraphMemo, topLevel, targetOptions, out var solver, out var opNodeMemo, out var tileNodeMemo, out var tileableNodeMemo);

        // 0. the top level already store a buffer at outter most.
        var toplevelStoreBufferConstraints = new List<Constraint>();
        var (inputBids, outputBids) = bufferGraphMemo[primTree.Wrapped].GetInputsOutputs();
        foreach (var (bid, binfo) in tileNodeMemo[primTree].BufferInfoMap)
        {
            if (inputBids.Contains(bid) || outputBids.Contains(bid))
            {
                var cons = solver.MakeEquality(binfo.Places[0][^1], 1);
                cons.SetName($"{bid}StoreAtOutMost");
                solver.Add(cons);
                toplevelStoreBufferConstraints.Add(cons);
            }
        }

        // 1. must have one buffer at lowest store level.
        // Beside the top-level node, from bottom to top count each tile node's buffer numbers which are stored at the lowest level.
        var tileNodeStoreAtLevelPlaces = new Dictionary<TileNode, Dictionary<BufferIdentity, Dictionary<int, List<IntExpr>>>>();
        var reusedBuffers = new HashSet<NodeWithBuffer>();
        primTree.Walk(
            treeNode =>
            {
                if (treeNode is not TileNode tileNode)
                {
                    return;
                }

                var tileNodeInfo = tileNodeMemo[tileNode];

                if (!tileNodeStoreAtLevelPlaces.TryGetValue(tileNode, out var curNodeStoreAtLevelePlaces))
                {
                    curNodeStoreAtLevelePlaces = new Dictionary<BufferIdentity, Dictionary<int, List<IntExpr>>>();
                    reusedBuffers.UnionWith(tileNodeInfo.DefUseMap.Keys.Select(b => new NodeWithBuffer(tileNode, b)));
                    foreach (var (bid, bufferInfo) in tileNodeInfo.BufferInfoMap)
                    {
                        var levelPlaces = new Dictionary<int, List<IntExpr>>();

                        // collect current node‘s placements.
                        for (int sl = 0; sl < tileNode.Level; sl++)
                        {
                            if (!levelPlaces.TryGetValue(sl, out var places))
                            {
                                places = new List<IntExpr>();
                                levelPlaces.Add(sl, places);
                            }

                            foreach (var place in bufferInfo.Places)
                            {
                                if (sl < place.Length)
                                {
                                    places.Add(place[sl]);
                                }
                            }
                        }

                        // collect child node's placement
                        foreach (var childNode in tileNode.Children.ToArray().OfType<TileNode>())
                        {
                            var childNodeStoreAtLevelePlaces = tileNodeStoreAtLevelPlaces[childNode];
                            if (tileNodeInfo.DefUseMap.ContainsKey(bid) || tileNodeInfo.DefUseMap.ContainsValue(bid))
                            {
                                continue;
                            }

                            // collect the child buffer's placement which has not been reused.
                            if (childNodeStoreAtLevelePlaces.TryGetValue(bid, out var childLevelPlaces))
                            {
                                for (int sl = 0; sl < childNode.Level; sl++)
                                {
                                    levelPlaces[sl].AddRange(childLevelPlaces[sl]);
                                }
                            }
                        }

                        curNodeStoreAtLevelePlaces.Add(bid, levelPlaces);
                    }

                    tileNodeStoreAtLevelPlaces.Add(tileNode, curNodeStoreAtLevelePlaces);
                }
            },
            true);

        // sum(places[cl,bid,ci,sl], (cl, ci)) == 1
        var eachLevelStoreBufferNumsConstrains = new Dictionary<BufferIdentity, Constraint[]>();
        foreach (var (bid, bufferInfo) in tileNodeMemo[primTree].BufferInfoMap)
        {
            if (reusedBuffers.Contains(new NodeWithBuffer(primTree, bid)))
            {
                continue;
            }

            var levelPlaces = tileNodeStoreAtLevelPlaces[primTree][bid];
            var cons = new Constraint[primTree.Level];
            eachLevelStoreBufferNumsConstrains[bid] = cons;
            for (int sl = 0; sl < primTree.Level; sl++)
            {
                if (levelPlaces.TryGetValue(sl, out var places))
                {
                    cons[sl] = solver.MakeEquality(solver.MakeSum(places.Select(e => e.Var()).ToArray()), 1);
                    cons[sl].SetName($"store[{bid}, sl{sl}]");
                    solver.Add(cons[sl]);
                }
            }
        }

        var eachLevelStoreReusedBufferNumsConstrains = new Dictionary<NodeWithBuffer, Constraint[]>();
        foreach (var (tileNode, bid) in reusedBuffers)
        {
            var fusedLevel = tileNode.Level - 1;

            // child's places
            var producerSubPlaces = new List<IntExpr>();
            var consumerSubPlaces = new List<IntExpr>();
            var nodeInfo = tileNodeMemo[tileNode];
            var sourceId = bid;
            var sinkId = nodeInfo.DefUseMap[sourceId];
            foreach (var childNode in tileNode.Children.ToArray().OfType<TileNode>())
            {
                var childNodeInfo = tileNodeMemo[childNode];
                foreach (var (cbid, cbidInfo) in childNodeInfo.BufferInfoMap)
                {
                    if (cbid == sourceId)
                    {
                        producerSubPlaces.AddRange(tileNodeStoreAtLevelPlaces[childNode][cbid][fusedLevel - 1]);
                    }
                    else if (cbid == sinkId)
                    {
                        consumerSubPlaces.AddRange(tileNodeStoreAtLevelPlaces[childNode][cbid][fusedLevel - 1]);
                    }
                }
            }

            // 1. child consumer sub places == child producer sub places == 0
            var producerChildStoreNums = solver.MakeSum(producerSubPlaces.Select(e => e.Var()).ToArray());
            var consumerChildStoreNums = solver.MakeSum(consumerSubPlaces.Select(e => e.Var()).ToArray());
            var pcons = solver.MakeEquality(producerChildStoreNums, 0);
            pcons.SetName($"producer_store[{bid}, sl{fusedLevel}]");
            solver.Add(pcons);
            var ccons = solver.MakeEquality(consumerChildStoreNums, 0);
            ccons.SetName($"consumer_store[{bid}, sl{fusedLevel}]");
            solver.Add(ccons);

            // 2. all parent places == 0
            var parentPlaces = new List<IntExpr>();
            var nextNode = tileNode;
            while (nextNode.Parent is TileNode nextParent)
            {
                for (int sl = tileNode.Level - 1; sl < primTree.Level; sl++)
                {
                    parentPlaces.AddRange(tileNodeStoreAtLevelPlaces[nextNode][bid][sl]);
                }

                nextNode = nextParent;
            }

            var parentCons = solver.MakeEquality(solver.MakeSum(parentPlaces.Select(e => e.Var()).ToArray()), 0);
            parentCons.SetName($"fused_parent_store[{bid}]");
            solver.Add(parentCons);

            // 3. fused places == 1
            var fusedStoreNums = solver.MakeSum(tileNodeStoreAtLevelPlaces[tileNode][bid][fusedLevel - 1].Select(e => e.Var()).ToArray());
            var fcons = solver.MakeEquality(fusedStoreNums, 1);
            fcons.SetName($"fused_store[{bid}, sl{fusedLevel}]");
            solver.Add(fcons);
            eachLevelStoreReusedBufferNumsConstrains.Add(new NodeWithBuffer(tileNode, bid), new[] { pcons, ccons, parentCons, fcons });
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
        // 5.1. sum(place[cl,b,ci,sl]*size[cl,b,ci], sl), sl = [0,toplevel)
        var levelBufferSizes = new Dictionary<int, Dictionary<NodeWithBuffer, IntExpr>>();
        var levelBufferLifeness = new Dictionary<int, Dictionary<NodeWithBuffer, Tuple<int, int>>>();
        var levelBufferLifenessConstraints = new Dictionary<int, Constraint[]>();
        for (int sl = 0; sl < topLevel - 1; sl++)
        {
            // note currently there is a only one root
            var nodeBufferSizes = levelBufferSizes[sl] = new();
            var nodeBufferLiveness = levelBufferLifeness[sl] = new();
            var beginTime = int.MaxValue;
            var endTime = int.MinValue;

            foreach (var (tileNode, nodeInfo) in tileNodeMemo)
            {
                foreach (var (bid, bufferInfo) in nodeInfo.BufferInfoMap)
                {
                    var nodeBuffer = new NodeWithBuffer(tileNode, bid);
                    nodeBufferLiveness[nodeBuffer] = bufferInfo.Liveness;
                    beginTime = Math.Min(beginTime, bufferInfo.Liveness.Item1);
                    endTime = Math.Max(endTime, bufferInfo.Liveness.Item2);
                    var extents = new List<IntExpr>();
                    for (int ci = 0; ci < bufferInfo.Places.Length; ci++)
                    {
                        if (sl >= bufferInfo.Places[ci].Length)
                        {
                            continue;
                        }

                        extents.Add(solver.MakeProd(bufferInfo.Places[ci][sl], bufferInfo.SizeVars[ci]));
                    }

                    nodeBufferSizes[nodeBuffer] = extents.Skip(1).Aggregate(extents[0], solver.MakeSum);
                }
            }

            // Add constraints according to liveness.
            if (Diagnostics.DumpScope.Current.IsEnabled(Diagnostics.DumpFlags.Tiling))
            {
                DumpGantt(nodeBufferSizes, nodeBufferLiveness, primTree, sl);
            }

            var lastTimeStamp = new HashSet<NodeWithBuffer>();
            var constraints = new List<Constraint>();
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
                    cons.SetName($"capacity[sl{sl}, t{i}]");
                    solver.Add(cons);
                    constraints.Add(cons);
                    lastTimeStamp.Clear(); // update last stamp.
                    lastTimeStamp.UnionWith(curTimeStamp);
                }
            }

            levelBufferLifenessConstraints.Add(sl, constraints.ToArray());
        }

        // compute the cycles as objective
        IntExpr computeCycles = solver.MakeIntConst(0);

        // the l0 write and l1 read update by op node. l0 have no reads.
        foreach (var (opNode, opNodeInfo) in opNodeMemo)
        {
            var tnode = (TileNode)opNode.Parent!;
            var loopTrip = tileNodeMemo[tnode].TripCounts[^1];
            var kernelInfo = opNode.GetKernelInfo(targetOptions);

            // make inner dimension increase.
            // var noContiguous = new IntExpr[opNodeInfo.Shapes.Length];
            // for (int i = 0; i < opNodeInfo.Shapes.Length; i++)
            // {
            //     noContiguous[i] = opNode.BufferShapes[i][^1] - opNodeInfo.Shapes[i][^1];
            // }
            // IntExpr opCycles = solver.MakeIntConst(120); // note cycles should get from op.
            IntExpr opCycles = kernelInfo.GetComputeCycle(opNodeInfo.Shapes, solver, opNode.GetMicroKernelContext(targetOptions));
            computeCycles = solver.MakeSum(computeCycles, opCycles * loopTrip);

            // computeCycles = solver.MakeSum(computeCycles, noContiguous.Aggregate(opCycles, solver.MakeSum) * loopTrip);
        }

        // Because of the placement as a control of data movement, there is no need to pick the buffer carefully.
        var levelDataReads = Enumerable.Range(0, topLevel).Select(i => (IntExpr)solver.MakeIntConst(0)).ToArray();
        var levelDataWrites = Enumerable.Range(0, topLevel).Select(i => (IntExpr)solver.MakeIntConst(0)).ToArray();
        foreach (var (tileNode, nodeInfo) in tileNodeMemo)
        {
            var createLevel = tileNode.Level;
            var nodeWrites = Enumerable.Range(0, topLevel).Select(_ => new List<IntExpr>()).ToArray();
            var nodeReads = Enumerable.Range(0, topLevel).Select(_ => new List<IntExpr>()).ToArray();
            foreach (var (bid, bufferInfo) in nodeInfo.BufferInfoMap)
            {
                var binfo = bid.Node.GetKernelInfo(targetOptions).BufferInfos;
                var reused = nodeInfo.DefUseMap.ContainsKey(bid);
                for (int storeLevel = 0; storeLevel < Math.Min(tileNode.Level, topLevel - 1); storeLevel++)
                {
                    // skip the buffer which store at top level
                    var volumes = Enumerable.Repeat((IntExpr)solver.MakeIntConst(1), bufferInfo.Places.Length).ToArray();
                    for (int i = 0; i < bufferInfo.Places.Length; i++)
                    {
                        if (storeLevel >= bufferInfo.Places[i].Length)
                        {
                            continue;
                        }

                        IntExpr factor = solver.MakeIntConst(1);

                        // if (storeLevel == 0 && !bid.IsOutput)
                        // {
                        //     // var elemSize = bid.Node.Grid.Buffers[bid.Index].CheckedDataType.SizeInBytes;
                        //     factor = bid.Node.BufferShapes[bid.Index][^1] - bufferInfo.Shapes[i][^1];
                        //     // var v = solver.MakeIsLessCstVar(width, 128) + 1; // for cache line.
                        // }
                        volumes[i] = bufferInfo.Places[i][storeLevel] * nodeInfo.TripCounts[i] * bufferInfo.SizeVars[i] * factor;

                        // volumes[i] = bufferInfo.Places[i][storeLevel] * bufferInfo.Trips[i] * bufferInfo.SizeVars[i] * factor;
                    }

                    var dataMoves = volumes.Skip(1).Aggregate(volumes[0], solver.MakeSum);

                    if (binfo[bid.Index].State.HasFlag(MicroKernelBufferInfo.BufferState.Read))
                    {
                        if (storeLevel < topLevel)
                        {
                            nodeWrites[storeLevel].Add(dataMoves); // write to store level.
                        }

                        if (storeLevel + 1 < topLevel && !reused)
                        {
                            nodeReads[storeLevel + 1].Add(dataMoves); // read from higher level.
                        }
                    }

                    // todo the intermediate buffer should be read write.
                    if (binfo[bid.Index].State.HasFlag(MicroKernelBufferInfo.BufferState.Write))
                    {
                        if (storeLevel + 1 < topLevel && !reused)
                        {
                            nodeWrites[storeLevel + 1].Add(dataMoves);
                        }

                        if (storeLevel < topLevel)
                        {
                            nodeReads[storeLevel].Add(dataMoves);
                        }
                    }
                }
            }

            for (int l = 0; l < topLevel; l++)
            {
                if (nodeWrites[l].Any())
                {
                    levelDataWrites[l] = levelDataWrites[l] + nodeWrites[l].Skip(1).Aggregate(nodeWrites[l].First(), solver.MakeSum);
                }

                if (nodeReads[l].Any())
                {
                    levelDataReads[l] = levelDataReads[l] + nodeReads[l].Skip(1).Aggregate(nodeReads[l].First(), solver.MakeSum);
                }
            }
        }

        var memoryCycles = new IntExpr[topLevel];
        for (int i = 0; i < topLevel; i++)
        {
            memoryCycles[i] = (levelDataWrites[i] + levelDataReads[i]).CeilDiv(memoryBandWidths[i]);
        }

        IntExpr totalCycles = computeCycles;
        for (int i = 0; i < topLevel; i++)
        {
            totalCycles = totalCycles + memoryCycles[i];
        }

        var totalCyclesVar = totalCycles.Var();
        totalCyclesVar.SetRange(1, long.MaxValue / memoryBandWidths[0]); /* avoid crash. */

        var objectiveMonitor = solver.MakeMinimize(totalCyclesVar, 1);
        var collector = solver.MakeNBestValueSolutionCollector(5, false);
        collector.AddObjective(totalCyclesVar);
        collector.Add(totalCyclesVar);
        collector.Add(levelDataReads.Select(i => i.Var()).ToArray());
        collector.Add(levelDataWrites.Select(i => i.Var()).ToArray());
        collector.Add(memoryCycles.Select(i => i.Var()).ToArray());
        collector.Add(computeCycles.Var());
        var searchAbleVars = new List<IntVar>();
        foreach (var (node, diminfo) in tileableNodeMemo)
        {
            searchAbleVars.AddRange(diminfo.TileVars.Select(i => i.Var()).Reverse());
            collector.Add(diminfo.TileVars.Select(i => i.Var()).ToArray());
            collector.Add(diminfo.ForwardExtents.Select(x => x.Var()).ToArray());
        }

        foreach (var (node, info) in opNodeMemo)
        {
            collector.Add(info.Shapes.SelectMany(i => i).Select(i => i.Var()).ToArray());
            collector.Add(info.Sizes.Select(i => i.Var()).ToArray());
        }

        foreach (var (node, info) in tileNodeMemo)
        {
            collector.Add(info.TripCounts.Select(i => i.Var()).ToArray());
            collector.Add(info.BackWardExtents.Select(i => i.Select(j => j.Var())).SelectMany(i => i).ToArray());
            foreach (var (bid, bufferInfo) in info.BufferInfoMap)
            {
                var placeVars = bufferInfo.Places.SelectMany(i => i).ToArray();
                searchAbleVars.AddRange(placeVars.Select(i => i.Var()));
                collector.Add(placeVars.Select(i => i.Var()).ToArray());
                collector.Add(bufferInfo.Shapes.SelectMany(i => i).Select(i => i.Var()).ToArray());
                collector.Add(bufferInfo.SizeVars.Where(v => v is not null).Select(i => i.Var()).ToArray());
                collector.Add(bufferInfo.SizeExprs.Where(v => v is not null).Select(i => i.Var()).ToArray());
                collector.Add(bufferInfo.Trips.Where(v => v is not null).Select(i => i.Var()).ToArray());
            }
        }

        foreach (var (_, nodeBufferSizes) in levelBufferSizes)
        {
            foreach (var (_, bufferSize) in nodeBufferSizes)
            {
                collector.Add(bufferSize.Var());
            }
        }

        foreach (var (_, v) in levelBufferLifenessConstraints)
        {
            foreach (var item in v)
            {
                collector.Add(item.Var());
            }
        }

        var defaultPhaseParameters = new DefaultPhaseParameters();
        if (Diagnostics.DumpScope.Current.IsEnabled(Diagnostics.DumpFlags.Tiling))
        {
            defaultPhaseParameters.display_level = DefaultPhaseParameters.NORMAL;
        }
        else
        {
            defaultPhaseParameters.display_level = DefaultPhaseParameters.NONE;
        }

        var decisionBuilder = solver.MakeDefaultPhase(searchAbleVars.ToArray(), defaultPhaseParameters);
        var solve_max_time = 30;
        if (System.Environment.GetEnvironmentVariable("NNCASE_TILING_MAX_TIME") is string s_solve_max_time)
        {
            try
            {
                solve_max_time = int.Parse(s_solve_max_time);
            }
            catch (System.Exception)
            {
            }
        }

        var solve_max_solutions = 15;
        if (System.Environment.GetEnvironmentVariable("NNCASE_TILING_MAX_SOLUTIONS") is string s_solve_max_solutions)
        {
            try
            {
                solve_max_solutions = int.Parse(s_solve_max_solutions);
            }
            catch (System.Exception)
            {
            }
        }

        var monitors = new List<SearchMonitor>() { collector, objectiveMonitor, solver.MakeTimeLimit(solve_max_time * 1000) };
        if (solve_max_solutions > 0)
        {
            monitors.Add(solver.MakeSolutionsLimit(solve_max_solutions));
        }

        if (Diagnostics.DumpScope.Current.IsEnabled(Diagnostics.DumpFlags.Tiling))
        {
            monitors.Add(solver.MakeSearchLog(10000, totalCyclesVar));
        }

        var status = solver.Solve(decisionBuilder, monitors.ToArray());
        if (!status)
        {
            DumpAssgin(primTree, new TreeSolverPrinter(null, solver, opNodeMemo, tileNodeMemo, tileableNodeMemo, targetOptions), tileVarConstraints, eachLevelStoreBufferNumsConstrains, levelBufferSizes, levelDataReads, levelDataWrites, memoryCycles, totalCycles, totalCyclesVar);
            throw new InvalidOperationException("tiling solve failed!");
        }

        var sol = collector.Solution(collector.SolutionCount() - 1);

        var levelBufferSizesAssgin = levelBufferSizes.ToDictionary(kv => kv.Key, kv => kv.Value.ToDictionary(p => p.Key, p => sol.Value(p.Value.Var())));
        var opNodeMemoAssgin = opNodeMemo.ToDictionary(kv => kv.Key, kv => new OpNodeInfo<long>(kv.Value.Maps, sol.Value(kv.Value.Shapes), sol.Value(kv.Value.Sizes)));
        var tileNodeMemoAssgin = tileNodeMemo.ToDictionary(kv => kv.Key, kv => new TileNodeInfo<long>(sol.Value(kv.Value.TripCounts), sol.Value(kv.Value.BackWardExtents), kv.Value.DefUseMap, kv.Value.BufferInfoMap.ToDictionary(p => p.Key, p => new TileNodeBufferInfo<long>(p.Value.Liveness, p.Value.Map, sol.Value(p.Value.Places), sol.Value(p.Value.Shapes), sol.Value(p.Value.SizeVars), sol.Value(p.Value.SizeExprs), sol.Value(p.Value.Trips), p.Value.Masks))));
        var tileableNodeMemoAssgin = tileableNodeMemo.ToDictionary(kv => kv.Key, kv => new DomainInfo<long>(sol.Value(kv.Value.TileVars), sol.Value(kv.Value.ForwardExtents), kv.Value.DimsMap));

        if (Diagnostics.DumpScope.Current.IsEnabled(Diagnostics.DumpFlags.Tiling))
        {
            DumpAssgin(primTree, new TreeSolverPrinter(sol, solver, opNodeMemo, tileNodeMemo, tileableNodeMemo, targetOptions), tileVarConstraints, eachLevelStoreBufferNumsConstrains, levelBufferSizes, levelDataReads, levelDataWrites, memoryCycles, computeCycles, totalCyclesVar);

            DumpAssgin(primTree, new TreeSolverPythonPrinter(sol, solver, opNodeMemo, tileNodeMemo, tileableNodeMemo, targetOptions), tileVarConstraints, eachLevelStoreBufferNumsConstrains, levelBufferSizes, levelDataReads, levelDataWrites, memoryCycles, computeCycles, totalCyclesVar);
        }

        return new TreeSolveResult(bufferGraphMemo[primTree.Wrapped], sol.ObjectiveValue(), levelBufferSizesAssgin, levelBufferLifeness, opNodeMemoAssgin, tileNodeMemoAssgin, tileableNodeMemoAssgin, targetOptions, moduleKind);
    }

    public static void DumpAssgin(ITreeNode tree, TreeSolverPythonPrinter printer, Dictionary<OpNode, Constraint[]> tileVarConstraints, Dictionary<BufferIdentity, Constraint[]> lowestStoreBufferNumsConstrains, Dictionary<int, Dictionary<NodeWithBuffer, IntExpr>> levelBufferSizes, IntExpr[] levelDataReads, IntExpr[] levelDataWrites, IntExpr[] memoryCycles, IntExpr computeCycles, IntVar totalCycles)
    {
        using (var stream = Diagnostics.DumpScope.Current.OpenFile($"modeling.py"))
        {
            using var baseWriter = new StreamWriter(stream);
            using var writer = new System.CodeDom.Compiler.IndentedTextWriter(baseWriter, "  ");
            tree.Accept(printer, (null, writer));
        }
    }

    public static void DumpAssgin(ITreeNode tree, TreeSolverPrinter printer, Dictionary<OpNode, Constraint[]> tileVarConstraints, Dictionary<BufferIdentity, Constraint[]> eachLevelStoreBufferNumsConstrains, Dictionary<int, Dictionary<NodeWithBuffer, IntExpr>> levelBufferSizes, IntExpr[] levelDataReads, IntExpr[] levelDataWrites, IntExpr[] memoryCycles, IntExpr computeCycles, IntVar totalCycles)
    {
        using (var stream = Diagnostics.DumpScope.Current.OpenFile($"modeling.yaml"))
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

            writer.WriteLine("EachLevelStoreBufferNumsConstrains:");
            writer.Indent++;
            foreach (var (node, cons) in eachLevelStoreBufferNumsConstrains)
            {
                TreeSolverPrinter.WriteIntExprVector(writer, node.ToString(), cons, printer.Solution);
            }

            writer.Indent--;

            writer.WriteLine("LevelMemoryUsage:");
            {
                writer.Indent++;
                foreach (var (sl, nodeMemoryUsage) in levelBufferSizes)
                {
                    writer.WriteLine($"Level_{sl}:");
                    writer.Indent++;
                    foreach (var (node, usage) in nodeMemoryUsage)
                    {
                        TreeSolverPrinter.WriteIntExpr(writer, $"- {node}", usage, printer.Solution);
                    }

                    writer.Indent--;
                }

                writer.Indent--;
            }

            TreeSolverPrinter.WriteIntExprVector(writer, "LevelDataReads", levelDataReads, printer.Solution);
            TreeSolverPrinter.WriteIntExprVector(writer, "LevelDataWrites", levelDataWrites, printer.Solution);
            TreeSolverPrinter.WriteIntExprVector(writer, "MemoryCycles", memoryCycles, printer.Solution);
            TreeSolverPrinter.WriteIntExpr(writer, "ComputeCycles", computeCycles, printer.Solution);
            TreeSolverPrinter.WriteIntExpr(writer, "TotalCycles", totalCycles, printer.Solution);
        }
    }

    public (Dictionary<TieredTileGraph, Expr> ResultMemo, long ObjectValue) SolveRootGraph(TieredTileGraph rootGraph, string moduleKind, ICpuTargetOptions targetOptions)
    {
        // bufferize root graph.
        var bufferGraphMemo = rootGraph.Bufferize();
        if (Diagnostics.DumpScope.Current.IsEnabled(Diagnostics.DumpFlags.Tiling))
        {
            bufferGraphMemo[rootGraph].Dump($"tile_buffer_graph");
        }

        // condense the root graph.
        var condensedGraph = rootGraph.Condense();
        if (Diagnostics.DumpScope.Current.IsEnabled(Diagnostics.DumpFlags.Tiling))
        {
            using (var file = Diagnostics.DumpScope.Current.OpenFile($"condensed_tile_graph.dot"))
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
        long objectValue = 0;
        foreach (var (primGraph, i) in condensedGraph.TopologicalSort().Select((s, i) => (s, i)))
        {
            using var subSubScope = new Diagnostics.DumpScope($"device_func_{DeviceFuncionCount}", Diagnostics.DumpFlags.Tiling);
            var primTree = treeGraphMemo[primGraph];
            HashSet<BufferIdentity> inputBids;
            HashSet<BufferIdentity> outputBids;

            if (!SolveMemo.TryGetValue(primTree, out var memo))
            {
                var result = SolvePrimGraph(primTree, bufferGraphMemo, targetOptions, moduleKind);
                (inputBids, outputBids) = (result.Inputs, result.Outputs);
                result.ScheduleBuffers();
                var bodyBuilder = T.Sequential();
                result.Visit(primTree, new(bodyBuilder, Array.Empty<Expr>()));
                var parameters = inputBids.Concat(outputBids).Select(k => result.PrimBufferMemo[k]).ToArray();
                var funcBuilder = T.PrimFunc($"device_func_{DeviceFuncionCount++}", moduleKind, parameters).Body(bodyBuilder);
                var primFunc = funcBuilder.Build();
                memo = new(new PrimFunctionWrapper(primFunc, inputBids.Count, inputBids.Concat(outputBids).Select(bid => bid.Node.Grid.GetArgument(bid.Index).CheckedType).ToArray()), result.ObjectiveValue);
                SolveMemo.Add(primTree, memo);
            }
            else
            {
                (inputBids, outputBids) = bufferGraphMemo[primGraph].GetInputsOutputs();
            }

            objectValue += memo.ObjectValue;
            var finalCall = new Call(memo.Func, inputBids.Select(bid => argumentsMemo[bid]).ToArray());
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

        return (resultMemo, objectValue);
    }

    public Expr Tile(Expr preExpr, string moduleKind, ICpuTargetOptions targetOptions)
    {
#if true
        var topLevel = targetOptions.MemoryCapacities.Length;
        var rootGraph = GraphBuilder.Build(preExpr, topLevel, out var exprMemo);
        if (Diagnostics.DumpScope.Current.IsEnabled(Diagnostics.DumpFlags.Tiling))
        {
            rootGraph.Dump($"tile_graph");
        }

        var (resultMemo, _) = SolveRootGraph(rootGraph, moduleKind, targetOptions);
        var cloner = new ReplacingExprCloner(exprMemo.ToDictionary(kv => (Expr)kv.Key, kv => resultMemo[kv.Value]));
        return cloner.Clone(preExpr, default);
#else
        var topLevel = targetOptions.MemoryCapacities.Length;
        var rootGraph = GraphBuilder.Build(preExpr, topLevel, out var exprMemo);
        var rootState = new MCTState(rootGraph, moduleKind, "0", SolveMemo, targetOptions);
        var rootNode = new MCTNode(rootState);
        var searcher = new MCTSearcher();
        searcher.Search(rootNode);
        if (Diagnostics.DumpScope.Current.IsEnabled(Diagnostics.DumpFlags.Tiling))
        {
            rootNode.Dump("SearchTree");
        }

        var bestState = (MCTState)searcher.BestMCTNode!.State;
        var replaces = new Dictionary<Expr, Expr>();
        foreach (var (oldExpr, v) in exprMemo)
        {
            if (bestState.Results.TryGetValue(v, out var newExpr))
            {
                replaces.Add(oldExpr, newExpr);
            }
        }

        var cloner = new ReplacingExprCloner(replaces);
        return cloner.Clone(preExpr, default);
#endif
    }

    private static void DumpGantt(Dictionary<NodeWithBuffer, IntExpr> nodeBufferSizes, Dictionary<NodeWithBuffer, Tuple<int, int>> nodeBufferLiveness, TileNode primTree, int storeLevel)
    {
        string GetStartStr(string name, int start) => $"[{name}] starts D+{start}";
        string GetDurationStr(string name, int duration) => $"[{name}] requires {duration} days";
        using (var fs = Diagnostics.DumpScope.Current.OpenFile($"Op{primTree.OpId}_{primTree.Level}_store_{storeLevel}_gantt.md"))
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

    public sealed record TiledFunc(PrimFunctionWrapper Func, long ObjectValue)
    {
    }
}
