// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Reactive;
using Google.OrTools.ConstraintSolver;
using Microsoft.Extensions.DependencyInjection;
using NetFabric.Hyperlinq;
using Nncase.Graphs;
using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.IR.Shapes;
using Nncase.Schedule.TileGraph;
using Nncase.TIR;
using QuikGraph;
using QuikGraph.Algorithms;
using QuikGraph.Graphviz;

namespace Nncase.Schedule;

public sealed class GraphTiler
{
    public Dictionary<TileNode, TiledFunc> SolveMemo { get; } = new Dictionary<TileNode, TiledFunc>(new ITreeNodeComparer());

    /// <summary>
    /// a simple cost model.
    /// </summary>
    public static TreeSolveResult SolvePrimGraph(TileNode primTree, Dictionary<TieredTileGraph, BufferGraph> bufferGraphMemo, INTTTargetOptions targetOptions, string moduleKind)
    {
        int[] memCapacities = targetOptions.MemoryCapacities;
        int[] memBandWidths = targetOptions.MemoryBandWidths;
        var levelCount = memCapacities.Length - 1;
        TreeSolverInitializer.Init(primTree, bufferGraphMemo, levelCount, targetOptions, out var solver, out var opNodeMemo, out var tileNodeMemo, out var tileableNodeMemo);

        // 0. each level buffer store at last accessed loop.
        var eachLevelStoreBufferConstrains = new Dictionary<int, Constraint[]>();
        for (int level = 0; level < levelCount; level++)
        {
            var cons = new List<Constraint>();
            foreach (var (tileNode, nodeInfo) in tileNodeMemo.Where(kv => kv.Key.Level == level))
            {
                foreach (var (bid, bufferInfo) in nodeInfo.BufferInfoMap)
                {
                    var pos = bufferInfo.GetLastRelatedPos();
                    for (int ci = 0; ci < bufferInfo.Places.Length; ci++)
                    {
                        for (int sl = 0; sl < bufferInfo.Places[ci].Length; sl++)
                        {
                            if (ci == pos && sl == level)
                            {
                                var c = solver.MakeEquality(bufferInfo.Places[ci][sl], 1);
                                c.SetName($"store[{tileNode}_{bid}_cl{ci}_sl{sl}]");
                                solver.Add(c);
                                cons.Add(c);
                            }
                            else
                            {
                                var c = solver.MakeEquality(bufferInfo.Places[ci][sl], 0);
                                c.SetName($"n_store[{tileNode}_{bid}_cl{ci}_sl{sl}]");
                                solver.Add(c);
                                cons.Add(c);
                            }
                        }
                    }
                }
            }

            eachLevelStoreBufferConstrains.Add(level, cons.ToArray());
        }

        // 1. tile var constraints
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
        var levelBufferShapes = new Dictionary<int, Dictionary<NodeWithBuffer, IntExpr[]>>();
        var levelBufferLifeness = new Dictionary<int, Dictionary<NodeWithBuffer, Tuple<int, int>>>();
        var levelBufferLifenessConstraints = new Dictionary<int, Constraint[]>();
        for (int sl = 0; sl < levelCount; sl++)
        {
            var nodeBufferSizes = levelBufferSizes[sl] = new();
            var nodeBufferLiveness = levelBufferLifeness[sl] = new();
            var nodeBufferShapes = levelBufferShapes[sl] = new();
            var beginTime = int.MaxValue;
            var endTime = int.MinValue;

            foreach (var (tileNode, nodeInfo) in tileNodeMemo.Where(kv => kv.Key.Level == sl)) // only consider create and store at same level.
            {
                foreach (var (bid, bufferInfo) in nodeInfo.BufferInfoMap)
                {
                    var nodeBuffer = new NodeWithBuffer(tileNode, bid);
                    var ci = bufferInfo.GetLastRelatedPos();
                    var extents = new List<IntExpr>();
                    beginTime = Math.Min(beginTime, bufferInfo.Liveness[ci].Item1);
                    endTime = Math.Max(endTime, bufferInfo.Liveness[ci].Item2);

                    extents.Add(solver.MakeProd(bufferInfo.Places[ci][sl], bufferInfo.Sizes[ci]));
                    var cons = solver.MakeGreater(bufferInfo.Sizes[ci], 0);
                    solver.Add(cons);
                    nodeBufferSizes[nodeBuffer] = solver.MakeSum(extents);
                    nodeBufferShapes[nodeBuffer] = bufferInfo.Shapes[ci];
                    nodeBufferLiveness[nodeBuffer] = bufferInfo.Liveness[ci];
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
                    var bufSizes = curTimeStamp.Select(key => nodeBufferSizes[key]).ToArray();
                    var totalSize = solver.MakeSum(bufSizes);
                    var cons = solver.MakeLessOrEqual(totalSize, memCapacities[sl]);
                    cons.SetName($"capacity_le[sl{sl}, t{i}]");
                    solver.Add(cons);
                    constraints.Add(cons);

                    // note can't determine the memory usage threshold.
                    // if (sl == 0)
                    // {
                    //     cons = solver.MakeGreaterOrEqual(totalSize, Math.Min((long)(curTimeStamp.Select(k => k.MaxSize).Sum() * 0.95), (long)(memCapacities[sl] * 0.5)));
                    //     cons.SetName($"capacity_ge[sl{sl}, t{i}]");
                    //     solver.Add(cons);
                    //     constraints.Add(cons);
                    // }
                    lastTimeStamp.Clear(); // update last stamp.
                    lastTimeStamp.UnionWith(curTimeStamp);
                }
            }

            levelBufferLifenessConstraints.Add(sl, constraints.ToArray());
        }

        // when buffer is read, the data read from last level memory.
        // when buffer is write, the data write to current level memory.
        var levelDataReads = Enumerable.Range(0, memCapacities.Length).Select(i => (IntExpr)solver.MakeIntConst(0)).ToArray();
        var levelDataWrites = Enumerable.Range(0, memCapacities.Length).Select(i => (IntExpr)solver.MakeIntConst(0)).ToArray();
        foreach (var (tileNode, nodeInfo) in tileNodeMemo)
        {
            var nodeWrites = Enumerable.Range(0, memCapacities.Length).Select(_ => new List<IntExpr>()).ToArray();
            var nodeReads = Enumerable.Range(0, memCapacities.Length).Select(_ => new List<IntExpr>()).ToArray();
            foreach (var (bid, bufferInfo) in nodeInfo.BufferInfoMap)
            {
                var binfo = bid.Node.GetKernelInfo(targetOptions).BufferInfos;
                var reused = nodeInfo.DefUseMap.ContainsKey(bid);
                for (int sl = 0; sl <= tileNode.Level; sl++)
                {
                    // skip the buffer which store at top level
                    var volume = (IntExpr)solver.MakeIntConst(1);
                    var ci = bufferInfo.GetLastRelatedPos();
                    IntExpr factor = solver.MakeIntConst(1); // todo factor for contiguous load/store.
                    volume = bufferInfo.Places[ci][sl] * bufferInfo.Trips[ci] * bufferInfo.Sizes[ci];

                    if (binfo[bid.Index].State.HasFlag(MicroKernelBufferInfo.BufferState.Read))
                    {
                        nodeReads[sl + 1].Add(volume); // read from last level.
                    }

                    // todo the intermediate buffer should be read write.
                    if (binfo[bid.Index].State.HasFlag(MicroKernelBufferInfo.BufferState.Write))
                    {
                        nodeWrites[sl + 1].Add(volume);
                    }
                }
            }

            for (int l = 0; l < memCapacities.Length; l++)
            {
                if (nodeWrites[l].Any())
                {
                    levelDataWrites[l] = levelDataWrites[l] + solver.MakeSum(nodeWrites[l]);
                }

                if (nodeReads[l].Any())
                {
                    levelDataReads[l] = levelDataReads[l] + solver.MakeSum(nodeReads[l]);
                }
            }
        }

        var memoryCycles = new IntExpr[memCapacities.Length];
        for (int i = 0; i < memCapacities.Length; i++)
        {
            memoryCycles[i] = (levelDataWrites[i] + levelDataReads[i]).CeilDiv(memBandWidths[i]);
        }

        IntExpr computeCycles = solver.MakeIntConst(10000);
        foreach (var (tileNode, nodeInfo) in tileNodeMemo.Where(kv => kv.Key.Level == 0))
        {
            computeCycles = computeCycles * nodeInfo.TripCounts[^1];
            break;
        }

        var totalCycles = (IntExpr)computeCycles;
        for (int i = 0; i < memCapacities.Length; i++)
        {
            totalCycles += memoryCycles[i];
        }

        var totalCyclesVar = totalCycles.Var();
        totalCyclesVar.SetRange(1, long.MaxValue / memBandWidths[0]); /* avoid crash. */
        var objectiveMonitor = solver.MakeMinimize(totalCyclesVar, 1);
        var collector = solver.MakeNBestValueSolutionCollector(5, false);
        collector.AddObjective(totalCyclesVar);
        collector.Add(totalCyclesVar);
        collector.Add(levelDataReads.Select(i => i.Var()).ToArray());
        collector.Add(levelDataWrites.Select(i => i.Var()).ToArray());
        collector.Add(computeCycles.Var());
        collector.Add(memoryCycles.Select(i => i.Var()).ToArray());
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
                collector.Add(bufferInfo.Sizes.Where(v => v is not null).Select(i => i.Var()).ToArray());
                collector.Add(bufferInfo.Trips.Where(v => v is not null).Select(i => i.Var()).ToArray());
            }
        }

        foreach (var (level, nodeBufferSizes) in levelBufferSizes)
        {
            foreach (var (nodeBuffer, bufferSize) in nodeBufferSizes)
            {
                collector.Add(bufferSize.Var());

                foreach (var shape in levelBufferShapes[level][nodeBuffer])
                {
                    collector.Add(shape.Var());
                }
            }
        }

        foreach (var (_, v) in levelBufferLifenessConstraints)
        {
            foreach (var item in v)
            {
                collector.Add(item.Var());
            }
        }

        // var defaultPhaseParameters = new DefaultPhaseParameters();
        // var decisionBuilder = solver.MakeDefaultPhase(searchAbleVars.ToArray(), defaultPhaseParameters);
        DecisionBuilder decisionBuilder;
        {
            var phaseTileVars = new List<IntVar>();
            var phaseOtherVars = new List<IntVar>();
            foreach (var (node, info) in opNodeMemo)
            {
                foreach (var tilevar in tileableNodeMemo[node].TileVars.Reverse())
                {
                    if (searchAbleVars.Contains(tilevar.Var()))
                    {
                        phaseTileVars.Add(tilevar.Var());
                    }
                }
            }

            phaseOtherVars.AddRange(searchAbleVars.Except(phaseTileVars));
            var phaseTiles = solver.MakePhase(phaseTileVars.ToArray(), Solver.CHOOSE_FIRST_UNBOUND, Solver.ASSIGN_MAX_VALUE);
            var phaseOthers = solver.MakePhase(phaseOtherVars.ToArray(), Solver.CHOOSE_FIRST_UNBOUND, Solver.ASSIGN_MIN_VALUE);
            decisionBuilder = solver.Compose(phaseTiles, phaseOthers);
        }

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
            DumpAssgin(primTree, new TreeSolverPrinter(null, solver, opNodeMemo, tileNodeMemo, tileableNodeMemo, targetOptions), tileVarConstraints, eachLevelStoreBufferConstrains, levelBufferLifenessConstraints, levelBufferSizes, levelDataReads, levelDataWrites, memoryCycles, computeCycles, totalCyclesVar);
            throw new SolveFailedException("tiling solve failed!");
        }

        var sol = collector.Solution(collector.SolutionCount() - 1);

        var levelBufferSizesAssgin = levelBufferSizes.ToDictionary(kv => kv.Key, kv => kv.Value.ToDictionary(p => p.Key, p => sol.Value(p.Value.Var())));
        var levelBufferInfos = new Dictionary<int, Dictionary<NodeWithBuffer, NodeWithBufferInfo>>();
        foreach (var (level, nodeBufferSizes) in levelBufferSizes)
        {
            var nodeBufferInfos = new Dictionary<NodeWithBuffer, NodeWithBufferInfo>();
            foreach (var (nodeBuffer, sizeVar) in nodeBufferSizes)
            {
                var liveness = levelBufferLifeness[level][nodeBuffer];
                var shapes = levelBufferShapes[level][nodeBuffer].Select(s => sol.Value(s.Var())).ToArray();
                var strides = TensorUtilities.GetDefaultStrides(shapes);
                nodeBufferInfos[nodeBuffer] = new NodeWithBufferInfo(sol.Value(sizeVar.Var()), liveness, shapes, strides);
            }

            levelBufferInfos[level] = nodeBufferInfos;
        }

        var opNodeMemoAssgin = opNodeMemo.ToDictionary(kv => kv.Key, kv => new OpNodeInfo<long>(kv.Value.Maps, sol.Value(kv.Value.Shapes), sol.Value(kv.Value.Sizes)));
        var tileNodeMemoAssgin = tileNodeMemo.ToDictionary(kv => kv.Key, kv => new TileNodeInfo<long>(sol.Value(kv.Value.TripCounts), sol.Value(kv.Value.BackWardExtents), kv.Value.DefUseMap, kv.Value.BufferInfoMap.ToDictionary(p => p.Key, p => new TileNodeBufferInfo<long>(p.Value.Liveness, p.Value.Map, sol.Value(p.Value.Places), sol.Value(p.Value.Shapes), sol.Value(p.Value.Sizes), sol.Value(p.Value.Trips), p.Value.Mask))));
        var tileableNodeMemoAssgin = tileableNodeMemo.ToDictionary(kv => kv.Key, kv => new DomainInfo<long>(sol.Value(kv.Value.TileVars), sol.Value(kv.Value.ForwardExtents), kv.Value.DimsMap));

        if (Diagnostics.DumpScope.Current.IsEnabled(Diagnostics.DumpFlags.Tiling))
        {
            DumpAssgin(primTree, new TreeSolverPrinter(sol, solver, opNodeMemo, tileNodeMemo, tileableNodeMemo, targetOptions), tileVarConstraints, eachLevelStoreBufferConstrains, levelBufferLifenessConstraints, levelBufferSizes, levelDataReads, levelDataWrites, memoryCycles, computeCycles, totalCyclesVar);

            DumpAssgin(primTree, new TreeSolverPythonPrinter(sol, solver, opNodeMemo, tileNodeMemo, tileableNodeMemo, targetOptions), tileVarConstraints, eachLevelStoreBufferConstrains, levelBufferLifenessConstraints, levelBufferSizes, levelDataReads, levelDataWrites, memoryCycles, computeCycles, totalCyclesVar);
        }

        return new TreeSolveResult(bufferGraphMemo[primTree.Wrapped], sol.ObjectiveValue(), levelBufferInfos, opNodeMemoAssgin, tileNodeMemoAssgin, tileableNodeMemoAssgin, targetOptions, moduleKind);
    }

    public static void DumpAssgin(ITreeNode tree, TreeSolverPythonPrinter printer, Dictionary<OpNode, Constraint[]> tileVarConstraints, Dictionary<int, Constraint[]> lowestStoreBufferNumsConstrains, Dictionary<int, Constraint[]> levelBufferLifenessConstraints, Dictionary<int, Dictionary<NodeWithBuffer, IntExpr>> levelBufferSizes, IntExpr[] levelDataReads, IntExpr[] levelDataWrites, IntExpr[] memoryCycles, IntExpr computeCycles, IntVar totalCycles)
    {
        using (var stream = Diagnostics.DumpScope.Current.OpenFile($"modeling.py"))
        {
            using var baseWriter = new StreamWriter(stream);
            using var writer = new System.CodeDom.Compiler.IndentedTextWriter(baseWriter, "  ");
            tree.Accept(printer, (null, writer));
        }
    }

    public static void DumpAssgin(ITreeNode tree, TreeSolverPrinter printer, Dictionary<OpNode, Constraint[]> tileVarConstraints, Dictionary<int, Constraint[]> eachLevelStoreBufferNumsConstrains, Dictionary<int, Constraint[]> levelBufferLifenessConstraints, Dictionary<int, Dictionary<NodeWithBuffer, IntExpr>> levelBufferSizes, IntExpr[] levelDataReads, IntExpr[] levelDataWrites, IntExpr[] memoryCycles, IntExpr computeCycles, IntVar totalCycles)
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

            writer.WriteLine("EachLevelBufferLifenessConstraints:");
            writer.Indent++;
            foreach (var (node, cons) in levelBufferLifenessConstraints)
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

    public (Dictionary<BufferIdentity, Expr> ArgumentMemo, long ObjectValue) SolveRootGraph(TieredTileGraph rootGraph, string moduleKind, INTTTargetOptions targetOptions, DimVar[] dynamicDimVars)
    {
        if (Diagnostics.DumpScope.Current.IsEnabled(Diagnostics.DumpFlags.Tiling))
        {
            rootGraph.Dump($"root_tile_graph");
        }

        // bufferize root graph.
        var bufferGraphMemo = rootGraph.Bufferize();
        if (Diagnostics.DumpScope.Current.IsEnabled(Diagnostics.DumpFlags.Tiling))
        {
            bufferGraphMemo[rootGraph].Dump($"root_buffer_graph");
        }

        // condense the root graph.
        var condensedGraph = rootGraph.Condense();
        if (Diagnostics.DumpScope.Current.IsEnabled(Diagnostics.DumpFlags.Tiling))
        {
            using (var file = Diagnostics.DumpScope.Current.OpenFile($"root_condensed_graph.dot"))
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

        var argumentMemo = bufferGraphMemo[rootGraph].GetInputsOutputs(null).Inputs.ToDictionary(k => k, k => k.Node.Grid.GetArgument(k.Index));
        long objectValue = 0;
        foreach (var (primGraph, i) in condensedGraph.TopologicalSort().Select((s, i) => (s, i)))
        {
            var funcName = CompileSessionScope.GetCurrentThrowIfNull().GetRequiredService<INamingProvider>().GetName("device_func");
            using var subSubScope = new Diagnostics.DumpScope(funcName, Diagnostics.DumpFlags.Tiling);
            var primTree = treeGraphMemo[primGraph];
            HashSet<BufferIdentity> inputBids;
            HashSet<BufferIdentity> outputBids;

            if (!SolveMemo.TryGetValue(primTree, out var tiled))
            {
                var result = SolvePrimGraph(primTree, bufferGraphMemo, targetOptions, moduleKind);
                (inputBids, outputBids) = (result.Inputs, result.Outputs);
                var maxAlign = result.ScheduleBuffers();
                var bodyBuilder = T.Sequential();
                var initOffsets = Enumerable.Repeat(new DimConst(0), primTree.DomainBoundExprs.Length).ToArray();
                var initBounds = primTree.DomainBoundExprs.ToArray();
                result.Visit(primTree, new(bodyBuilder, initOffsets, initBounds));
                var parameters = inputBids.Select(k => result.InputOutputVars[k]).Concat(
                    dynamicDimVars.Select(v => (IVar)v.With())).Concat(
                    outputBids.Select(k => result.InputOutputVars[k])).ToArray();
                var funcBuilder = T.PrimFunc(funcName, moduleKind, parameters).Body(bodyBuilder);
                var primFunc = funcBuilder.Build();
                {
                    // note noneed to rewrite shapeof, because we don't use shapeof new.
                    // var gridBufferToVarMap = inputBids.Concat(outputBids).Select(bid => bid.Node.Grid.GetArgument(bid.Index)).Zip(parameters.Where(p => p is not DimVar)).ToDictionary(p => p.First, p => (Expr)p.Second, (IEqualityComparer<Expr>)ReferenceEqualityComparer.Instance);
                    // var mutator = new AtShapeOfRewriter(gridBufferToVarMap);
                    // mutator.Visit(primFunc, default);
                }

                primFunc.SchedResult.IsScheduled = true; // avoid buffersize pass schedule it again.
                primFunc.SchedResult.DataAlign = (ulong)maxAlign;
                tiled = new(new PrimFunctionWrapper(primFunc, inputBids.Count + dynamicDimVars.Length, inputBids.Select(bid => bid.Node.Grid.GetArgument(bid.Index).CheckedType).Concat(dynamicDimVars.Select(v => new DimensionType(DimensionKind.Dynamic))).Concat(outputBids.Select(bid => bid.Node.Grid.GetArgument(bid.Index).CheckedType)).ToArray()), result.ObjectiveValue);
                SolveMemo.Add(primTree, tiled);
            }
            else
            {
                (inputBids, outputBids) = bufferGraphMemo[primGraph].GetInputsOutputs(bufferGraphMemo[rootGraph]);
            }

            objectValue += tiled.ObjectValue;
            var finalCall = new Call(tiled.Func, inputBids.Select(bid => argumentMemo[bid]).Concat(dynamicDimVars.OfType<BaseExpr>()).ToArray());

            // save the output.
            foreach (var (outputBid, outputIndex) in outputBids.Select((b, i) => (b, i)))
            {
                if (!argumentMemo.TryGetValue(outputBid, out var _))
                {
                    var outputExpr = finalCall;

                    // process the tuple output.
                    if (outputBids.Count > 1)
                    {
                        outputExpr = IR.F.Tensors.GetItem(outputExpr, outputIndex);
                    }

                    argumentMemo.Add(outputBid, outputExpr);

                    // other prim graph's argument requires input bid, so we need to find it.
                    foreach (var sinkBid in bufferGraphMemo[rootGraph].OutEdges(outputBid).
                        Where(e => e.Tag is BufferEdgeKind.Inter).
                        Select(edge => edge.Target))
                    {
                        if (!argumentMemo.ContainsKey(sinkBid))
                        {
                            argumentMemo.Add(sinkBid, outputExpr);
                        }
                    }
                }
            }
        }

        return (argumentMemo, objectValue);
    }

    public BaseExpr Tile(BaseExpr preExpr, string moduleKind, INTTTargetOptions targetOptions, DimVar[] dynamicDimVars)
    {
        var levelCount = targetOptions.MemoryCapacities.Length - 1;
        var rootGraph = TieredTileGraphBuilder.Build(preExpr, levelCount, out var exprMemo);
#if false
        if (Diagnostics.DumpScope.Current.IsEnabled(Diagnostics.DumpFlags.Tiling))
        {
            rootGraph.Dump($"tile_graph");
        }

        var (resultMemo, _) = SolveRootGraph(rootGraph, moduleKind, targetOptions, dynamicDimVars);
        var cloner = new ReplacingExprCloner(exprMemo.ToDictionary(kv => (BaseExpr)kv.Key, kv => (BaseExpr)resultMemo[kv.Value]));
        return cloner.Clone(preExpr, default);
#else
        var rootState = new MCTState(rootGraph, moduleKind, "0", this, targetOptions, dynamicDimVars);
        var rootNode = new MCTNode(rootState);
        var visitTimes = 100u;
        if (System.Environment.GetEnvironmentVariable("NNCASE_TILING_MAX_VISIT") is string s_search_times)
        {
            try
            {
                visitTimes = uint.Parse(s_search_times);
            }
            catch (System.Exception)
            {
            }
        }

        var searcher = new MCTSearcher((int)visitTimes);
        searcher.Search(rootNode);
        if (Diagnostics.DumpScope.Current.IsEnabled(Diagnostics.DumpFlags.Tiling))
        {
            rootNode.Dump("SearchTree");
        }

        var bestState = (MCTState)searcher.BestMCTNode!.State;
        var replaces = new Dictionary<BaseExpr, BaseExpr>();

        foreach (var (bid, value) in bestState.ArgumentMemo)
        {
            // use bid to find the old expr.
            var oldExpr = bid.IsOutput ? bid.Node.Grid : bid.Node.Grid.GetArgument(bid.Index);
            if (!replaces.ContainsKey(oldExpr))
            {
                replaces.Add(oldExpr, value);
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

    private sealed class AtShapeOfRewriter : ExprRewriter
    {
        private readonly Dictionary<Expr, Expr> _exprMap;

        public AtShapeOfRewriter(Dictionary<Expr, Expr> exprMap)
            : base(false)
        {
            _exprMap = exprMap;
        }

        protected override BaseExpr VisitDimAt(DimAt at, Unit context)
        {
            if (at.Shape is ShapeOf { Value: Expr expr } && _exprMap.TryGetValue(expr, out var newExpr))
            {
                return new DimAt(new ShapeOf(newExpr), at.Index)
                {
                    Metadata = at.Metadata,
                };
            }

            return base.VisitDimAt(at, context);
        }
    }
}

internal sealed class SolveFailedException : Exception
{
    public SolveFailedException(string message)
        : base(message)
    {
    }
}
