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

    public Expr Tile(Expr preExpr, string moduleKind, string itemNumber, ICpuTargetOptions targetOptions)
    {
        var topLevel = targetOptions.MemoryCapacities.Length;
        var rootGraph = GraphBuilder.Build(preExpr, topLevel, out var exprMemo);
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
            using var subscope = new Diagnostics.DumpScope($"device_func{itemNumber}_{i}", Diagnostics.DumpFlags.Tiling);
            var primTree = treeGraphMemo[primGraph];
            var primBufferGraph = bufferGraphMemo[primGraph];
            HashSet<BufferIdentity> inputBids;
            HashSet<BufferIdentity> outputBids;

            if (!_primFuncMemo.TryGetValue(primTree, out var wrapper))
            {
                var result = SolvePrimGraph(primTree, primBufferGraph, targetOptions);
                (inputBids, outputBids) = (result.Inputs, result.Outputs);
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
                (inputBids, outputBids) = primBufferGraph.GetInputsOutputs();
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
        var topLevel = memoryCapacities.Length;
        TreeSolverInitializer.Init(primTree, topLevel, targetOptions, out var solver, out var opNodeMemo, out var tileNodeMemo, out var tileableNodeMemo);

        // 0. the top level already store a buffer at outter most.
        var toplevelStoreBufferConstraints = new List<Constraint>();
        foreach (var (bid, binfo) in tileNodeMemo[primTree].BufferInfoMap)
        {
            var cons = solver.MakeEquality(binfo.Places[0][^1], 1);
            cons.SetName($"{bid}StoreAtOutMost");
            solver.Add(cons);
            toplevelStoreBufferConstraints.Add(cons);
        }

        // 0.1 parent node's inner place is equal to child's outter place.
        var duplictePlaceConstranits = new List<Constraint>();
        primTree.Walk(
            (treeNode) =>
            {
                if (treeNode is not TileNode tileNode || tileNode.Level == 1)
                {
                    return;
                }

                foreach (var (bid, binfo) in tileNodeMemo[tileNode].BufferInfoMap)
                {
                    foreach (var child in tileNode.Children.ToArray().OfType<TileNode>())
                    {
                        var cbinfo = tileNodeMemo[child].BufferInfoMap[tileNodeMemo[child].GetCacheBid(bid)];
                        for (int sl = 0; sl < tileNode.Level - 1; sl++)
                        {
                            var cons = solver.MakeLessOrEqual(binfo.Places[^1][sl] + cbinfo.Places[0][sl], 1);
                            duplictePlaceConstranits.Add(cons);
                            solver.Add(cons);
                        }
                    }
                }
            });

        // 1. must have one buffer at lowest store level.
        // Beside the top-level node, from bottom to top count each tile node's buffer numbers which are stored at the lowest level.
        var eachLevelStoreBufferNums = new Dictionary<TileNode, Dictionary<BufferIdentity, Dictionary<int, IntExpr>>>();
        primTree.Walk(
            treeNode =>
            {
                if (treeNode is not TileNode tileNode)
                {
                    return;
                }

                var tileNodeInfo = tileNodeMemo[tileNode];

                if (!eachLevelStoreBufferNums.TryGetValue(tileNode, out var nodeStoreBufferNums))
                {
                    nodeStoreBufferNums = new Dictionary<BufferIdentity, Dictionary<int, IntExpr>>();
                    foreach (var (bid, bufferInfo) in tileNodeInfo.BufferInfoMap)
                    {
                        var levelStoreNums = new Dictionary<int, IntExpr>();
                        for (int sl = 0; sl < tileNode.Level; sl++)
                        {
                            levelStoreNums[sl] = solver.MakeSum(bufferInfo.Places.Select(p => p[sl].Var()).ToArray());
                        }

                        nodeStoreBufferNums.Add(bid, levelStoreNums);
                    }

                    foreach (var child in tileNode.Children.ToArray().OfType<TileNode>())
                    {
                        foreach (var (cbid, cbufferInfo) in tileNodeMemo[child].BufferInfoMap)
                        {
                            var pbid = tileNodeInfo.GetCacheBid(cbid);
                            for (int sl = 0; sl < child.Level; sl++)
                            {
                                nodeStoreBufferNums[pbid][sl] = nodeStoreBufferNums[pbid][sl] + eachLevelStoreBufferNums[child][cbid][sl];
                            }
                        }
                    }

                    eachLevelStoreBufferNums.Add(tileNode, nodeStoreBufferNums);
                }
            },
            true);

        var eachLevelStoreBufferNumsConstrains = new Dictionary<BufferIdentity, Constraint[]>();
        foreach (var (bid, bufferInfo) in tileNodeMemo[primTree].BufferInfoMap)
        {
            var cons = new Constraint[primTree.Level];
            eachLevelStoreBufferNumsConstrains[bid] = cons;
            for (int sl = 0; sl < primTree.Level; sl++)
            {
                cons[sl] = solver.MakeEquality(eachLevelStoreBufferNums[primTree][bid][sl], 1);
                cons[sl].SetName($"store[{bid}, sl{sl}]");
                solver.Add(cons[sl]);
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
        for (int sl = 0; sl < topLevel - 1; sl++)
        {
            // note currently there is a only one root
            var nodeBufferSizes = levelBufferSizes[sl] = new();
            var nodeBufferLiveness = levelBufferLifeness[sl] = new();
            var rootNodeInfo = tileNodeMemo[primTree];
            var beginTime = int.MaxValue;
            var endTime = int.MinValue;

            foreach (var (bid, bufferInfo) in rootNodeInfo.BufferInfoMap)
            {
                var extents = bufferInfo.Places.Select(p => p[sl]).Zip(bufferInfo.SizeVars).Select(p => p.First * p.Second).ToArray();
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

                if (current is not TileNode childNode || childNode.Level <= sl)
                {
                    return;
                }

                foreach (var (cbid, childBufferInfo) in tileNodeMemo[childNode].BufferInfoMap)
                {
                    // accumulate the extents
                    var extents = childBufferInfo.Places.Select(p => p[sl]).Zip(childBufferInfo.SizeVars).Select(p => p.First * p.Second).ToArray();
                    nodeBufferSizes[new(childNode, cbid)] = extents.Skip(1).Aggregate(extents[0], solver.MakeSum);
                    nodeBufferLiveness[new(childNode, cbid)] = childBufferInfo.Liveness;
                    beginTime = Math.Min(beginTime, childBufferInfo.Liveness.Item1);
                    endTime = Math.Max(endTime, childBufferInfo.Liveness.Item2);
                }
            });

            // Add constraints according to liveness.
#if false
            DumpGantt(nodeBufferSizes, nodeBufferLiveness, primTree, sl);
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

        // from top to down.
        var levelDataReads = Enumerable.Range(0, topLevel).Select(i => (IntExpr)solver.MakeIntConst(0)).ToArray();
        var levelDataWrites = Enumerable.Range(0, topLevel).Select(i => (IntExpr)solver.MakeIntConst(0)).ToArray();
        foreach (var (tileNode, nodeInfo) in tileNodeMemo)
        {
            var createLevel = tileNode.Level;
            var nodeWrites = Enumerable.Range(0, topLevel).Select(_ => new List<IntExpr>()).ToArray();
            var nodeReads = Enumerable.Range(0, topLevel).Select(_ => new List<IntExpr>()).ToArray();
            MicroKernelBufferInfo[]? binfo = null;
            foreach (var (bid, bufferInfo) in nodeInfo.BufferInfoMap)
            {
                binfo ??= bid.Node.GetKernelInfo(targetOptions).BufferInfos;
                for (int storeLevel = 0; storeLevel < bufferInfo.Places[0].Length; storeLevel++)
                {
                    var volumes = new IntExpr[bufferInfo.Places.Length];
                    for (int i = 0; i < bufferInfo.Places.Length; i++)
                    {
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
                        if (storeLevel < topLevel - 1)
                        {
                            nodeWrites[storeLevel].Add(dataMoves); // write to store level.
                        }

                        if (storeLevel < topLevel - 1)
                        {
                            nodeReads[storeLevel + 1].Add(dataMoves); // read from create level.
                        }
                    }

                    // todo the intermediate buffer should be read write.
                    if (binfo[bid.Index].State.HasFlag(MicroKernelBufferInfo.BufferState.Write))
                    {
                        if (storeLevel < topLevel - 1)
                        {
                            nodeWrites[storeLevel + 1].Add(dataMoves);
                        }

                        if (storeLevel < topLevel - 1)
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
            // memoryCycles[i] = (levelDataWrites[i] + levelDataReads[i]).CeilDiv(memoryBandWidths[i]);
            memoryCycles[i] = levelDataWrites[i].CeilDiv(memoryBandWidths[i]);
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
            searchAbleVars.AddRange(diminfo.TileVars.Select(i => i.Var()));
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
            foreach (var (bid, bufferInfo) in info.BufferInfoMap)
            {
                var placeVars = bufferInfo.Places.SelectMany(i => i).ToArray();
                searchAbleVars.AddRange(placeVars.Select(i => i.Var()));
                collector.Add(placeVars.Select(i => i.Var()).ToArray());
                collector.Add(bufferInfo.Shapes.SelectMany(i => i).Select(i => i.Var()).ToArray());
                collector.Add(bufferInfo.SizeVars.Select(i => i.Var()).ToArray());
                collector.Add(bufferInfo.SizeExprs.Select(i => i.Var()).ToArray());
                collector.Add(bufferInfo.Trips.Select(i => i.Var()).ToArray());
            }
        }

        foreach (var (_, nodeBufferSizes) in levelBufferSizes)
        {
            foreach (var (_, bufferSize) in nodeBufferSizes)
            {
                collector.Add(bufferSize.Var());
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
        var monitors = new List<SearchMonitor>() { collector, objectiveMonitor, /* solver.MakeSolutionsLimit(30), */ solver.MakeTimeLimit(50000) };
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

        return new TreeSolveResult(primBufferGraph, sol.ObjectiveValue(), levelBufferSizesAssgin, levelBufferLifeness, opNodeMemoAssgin, tileNodeMemoAssgin, tileableNodeMemoAssgin, targetOptions);
    }

    private void DumpAssgin(ITreeNode tree, TreeSolverPythonPrinter printer, Dictionary<OpNode, Constraint[]> tileVarConstraints, Dictionary<BufferIdentity, Constraint[]> lowestStoreBufferNumsConstrains, Dictionary<int, Dictionary<NodeWithBuffer, IntExpr>> levelBufferSizes, IntExpr[] levelDataReads, IntExpr[] levelDataWrites, IntExpr[] memoryCycles, IntExpr computeCycles, IntVar totalCycles)
    {
        using (var stream = Diagnostics.DumpScope.Current.OpenFile($"modeling.py"))
        {
            using var baseWriter = new StreamWriter(stream);
            using var writer = new System.CodeDom.Compiler.IndentedTextWriter(baseWriter, "  ");
            tree.Accept(printer, (null, writer));
        }
    }

    private void DumpAssgin(ITreeNode tree, TreeSolverPrinter printer, Dictionary<OpNode, Constraint[]> tileVarConstraints, Dictionary<BufferIdentity, Constraint[]> eachLevelStoreBufferNumsConstrains, Dictionary<int, Dictionary<NodeWithBuffer, IntExpr>> levelBufferSizes, IntExpr[] levelDataReads, IntExpr[] levelDataWrites, IntExpr[] memoryCycles, IntExpr computeCycles, IntVar totalCycles)
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

    private void DumpGantt(Dictionary<NodeWithBuffer, IntExpr> nodeBufferSizes, Dictionary<NodeWithBuffer, Tuple<int, int>> nodeBufferLiveness, TileNode primTree, int storeLevel)
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
}
