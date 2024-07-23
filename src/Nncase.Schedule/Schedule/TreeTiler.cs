// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using GiGraph.Dot.Extensions;
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

    public static void Dump(ITreeNode tree, string name)
    {
        using (var stream = Diagnostics.DumpScope.Current.OpenFile($"{name}.dot"))
        {
            using var writer = new StreamWriter(stream);
            var printer = new TreePrinter();
            tree.Accept(printer, TreePrinter.Context.Default);
            printer.Graph.Build(writer);
        }
    }

    public static bool Merge(ITreeNode tree, int opConsumer, int opProducer, int level)
    {
        var merger = new TreeMerger(opConsumer, opProducer, level);
        return tree.Accept(merger, default);
    }

    public static void DumpAssgin(ITreeNode tree, TreeSolverPrinter printer)
    {
        // using (var stream = Diagnostics.DumpScope.Current.OpenFile($"model.py"))
        // {
        //     using var baseWriter = new StreamWriter(stream);
        //     var printer = new TreeSolverPrinter(baseWriter, sol, solver, opNodeMemo, tileNodeMemo, tileableNodeMemo, compileOptions.TargetOptions);
        //     tree.Accept(printer, default);
        //     using var writer = new System.CodeDom.Compiler.IndentedTextWriter(baseWriter, "  ");
        //     writer.WriteLine("tileVarConstraints:");
        //     writer.Indent++;
        //     foreach (var (opnode, consts) in tileVarConstraints)
        //     {
        //         TreeSolverPrinter.WriteIntExprVector(writer, opnode.ToString(), consts, sol);
        //     }

        //     writer.Indent--;

        //     writer.WriteLine("lowestStoreBufferNumsConstrains:");
        //     writer.Indent++;
        //     foreach (var (node, cons) in lowestStoreBufferNumsConstrains)
        //     {
        //         TreeSolverPrinter.WriteIntExprVector(writer, node.ToString(), new[] { cons }, sol);
        //     }

        //     writer.Indent--;

        //     writer.WriteLine("EachParentNodeCreateBufferConstraints:");
        //     writer.Indent++;
        //     foreach (var (node, constraints) in eachParentNodeCreateBufferConstraints)
        //     {
        //         TreeSolverPrinter.WriteIntExprVector(writer, node.ToString(), constraints.Values.ToArray(), sol);
        //     }

        //     writer.Indent--;

        //     writer.WriteLine("memoryCapacityConstraints:");
        //     writer.Indent++;

        //     // for (int l = 1; l < totalLevel; l++)
        //     // {
        //     //     TreeSolverPrinter.WriteIntExprVector(writer, l.ToString(), levelNodeBufferOffset[l].ToArray(), sol);
        //     //     TreeSolverPrinter.WriteIntExprVector(writer, l.ToString(), levelNodeBufferExtent[l].ToArray(), sol);
        //     // }
        //     writer.Indent--;

        //     TreeSolverPrinter.WriteIntExprVector(writer, "levelDataReads:", levelDataReads, sol);
        //     TreeSolverPrinter.WriteIntExprVector(writer, "levelDataWrites:", levelDataWrites, sol);
        //     TreeSolverPrinter.WriteIntExprVector(writer, "memoryCycles:", memoryCycles, sol);
        //     writer.WriteLine($"computeCycles: {computeCycles.ToSimplifyString()}");
        // }
    }

    public static bool Solve(ITreeNode tree, int totalLevel, CompileOptions compileOptions, out TreeSolverResultConstructor resultConstructor)
    {
        long[] memoryCapacitys = new long[] { 2 * 1024 * 1024, int.MaxValue }; // l1, l2
        long[] memoryBandWidths = new long[] { 256, 128, 4 }; // l0, l1, l2
        var solver = new Solver("treeSolver");
        var opNodeMemo = new Dictionary<OpNode, OpNodeInfo>();
        var tileNodeMemo = new Dictionary<TileNode, TileNodeInfo>();
        var tileableNodeMemo = new Dictionary<ITileAbleNode, DomainInfo>();
        var init = new TreeSolverInitializer(totalLevel, solver, opNodeMemo, tileNodeMemo, tileableNodeMemo, compileOptions.TargetOptions);
        var argumentsInfo = TreeSolverInitializer.GetArgumentsInfo(tree.Accept(init, TreeSolverInitializer.Context.Default).BufferResults);
        var initWrites = new TreeSolverWritesInitializer(solver, opNodeMemo, tileNodeMemo, tileableNodeMemo, compileOptions.TargetOptions);
        tree.Accept(initWrites, new());

        // 1. each buffer must store one at lowest level.
        // 1.1 count each node's buffer store nums.
        var lowestStoreBufferNums = new Dictionary<TileNode, Dictionary<BufferIdenitity, IntExpr>>();
        foreach (var (tileNode, bufferInfoMemo) in tileNodeMemo)
        {
            var tileStoreNums = new Dictionary<BufferIdenitity, IntExpr>();
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
        var lowestStoreBufferNumsConstrains = new Dictionary<BufferIdenitity, Constraint>();
        foreach (var (node, bufferInfoMemo) in tileNodeMemo.Where(t => t.Key.Level == totalLevel))
        {
            foreach (var (bid, bufferInfo) in bufferInfoMemo.BufferInfoMap)
            {
                lowestStoreBufferNumsConstrains[bid] = solver.MakeEquality(lowestStoreBufferNums[node][bid], 1);
                solver.Add(lowestStoreBufferNumsConstrains[bid]);
            }
        }

        // 2. each tensor only can create one or zero buffer at each create level.
        var eachNodeCreateBufferConstraints = new Dictionary<TileNode, Dictionary<BufferIdenitity, Constraint>>();
        var eachNodeCreateBufferNums = new Dictionary<TileNode, Dictionary<BufferIdenitity, IntExpr>>();
        foreach (var (node, nodeInfo) in tileNodeMemo)
        {
            var createBufferConstraints = eachNodeCreateBufferConstraints[node] = new Dictionary<BufferIdenitity, Constraint>();
            var createBufferNums = eachNodeCreateBufferNums[node] = new Dictionary<BufferIdenitity, IntExpr>();
            foreach (var (bid, bufferInfo) in nodeInfo.BufferInfoMap)
            {
                createBufferNums[bid] = solver.MakeSum(bufferInfo.Places.SelectMany(i => i).ToArray());
                createBufferConstraints[bid] = solver.MakeLessOrEqual(createBufferNums[bid], 1);
                createBufferConstraints[bid].SetName($"nodeCreate[{node.Level}, {node.OpId}, {bid}]");
                solver.Add(createBufferConstraints[bid]);
            }
        }

        // 2.1 each cache buffer requires it's parent level create a buffer.
        var eachParentNodeCreateBufferConstraints = new Dictionary<TileNode, Dictionary<BufferIdenitity, Constraint>>();
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
        var levelNodeBufferBoxs = new Dictionary<int, Dictionary<TileNode, Dictionary<BufferIdenitity, IntExpr[]>>>();
        for (int sl = 1; sl < totalLevel; sl++)
        {
            var nodeBufferBoxs = levelNodeBufferBoxs[sl] = new();
            foreach (var (parentNode, parentNodeInfo) in tileNodeMemo.Where(p => p.Key.Level == totalLevel))
            {
                var bufferBoxs = nodeBufferBoxs[parentNode] = new();
                var banedMemo = new HashSet<BufferIdenitity>(parentNodeInfo.DefUseMap.Keys.Concat(parentNodeInfo.DefUseMap.Values));
                foreach (var (bid, bufferInfo) in parentNodeInfo.BufferInfoMap)
                {
                    var box = bufferBoxs[bid] = new IntExpr[4];

                    // x_var, x_size, y_var, y_size
                    box[0] = solver.MakeIntConst(bufferInfo.Lifeness.Item1, $"start[{sl}, {parentNode}, {bid}]");
                    box[1] = solver.MakeIntConst(bufferInfo.Lifeness.Item2 - bufferInfo.Lifeness.Item1);
                    box[2] = solver.MakeIntVar(0, memoryCapacitys[sl - 1] - 1, $"offset[{sl}, {parentNode}, {bid}]");
                    var extents = bufferInfo.Places.Select(p => p[sl - 1]).Zip(bufferInfo.SizeVars).Select(p => p.First * p.Second).ToArray();
                    box[3] = extents.Skip(1).Aggregate(extents[0], solver.MakeSum);
                }

                void AccumulateChildExtents(ITreeNode currnet)
                {
                    switch (currnet)
                    {
                        case ScopeNode scopeNode:
                            foreach (var child in scopeNode.Children)
                            {
                                AccumulateChildExtents(child);
                            }

                            break;
                        case TileNode { Level: >= 1 } childNode:
                            {
                                foreach (var (cbid, childBufferInfo) in tileNodeMemo[childNode].BufferInfoMap)
                                {
                                    if (childNode.Level == 1 && parentNode.Level == 2 && banedMemo.Contains(cbid))
                                    {
                                        continue;
                                    }

                                    // accumulate the extents
                                    var extents = childBufferInfo.Places.Select(p => p[sl - 1]).Zip(childBufferInfo.SizeVars).Select(p => p.First * p.Second).ToArray();
                                    bufferBoxs[cbid][3] += extents.Skip(1).Aggregate(extents[0], solver.MakeSum);
                                }

                                AccumulateChildExtents(childNode.Child);
                            }

                            break;
                        default:
                            break;
                    }
                }

                AccumulateChildExtents(parentNode.Child);

                var x_vars = bufferBoxs.Values.Select(box => box[0].Var()).ToArray();
                var x_sizes = bufferBoxs.Values.Select(box => box[1].Var()).ToArray();
                var y_vars = bufferBoxs.Values.Select(box => box[2].Var()).ToArray();
                var y_sizes = bufferBoxs.Values.Select(box => box[3].Var()).ToArray();

                solver.Add(solver.MakeSumLessOrEqual(y_sizes, memoryCapacitys[sl - 1]));

                // note can't schedule buffers.
                // solver.Add(solver.MakeNonOverlappingNonStrictBoxesConstraint(x_vars, y_vars, x_sizes, y_sizes));
                // foreach (var topOffset in y_vars.Zip(y_sizes).Select(p => p.First + p.Second))
                // {
                //     solver.Add(solver.MakeLessOrEqual(topOffset, memoryCapacitys[sl - 1]));
                // }
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
                        nodeWrites[sl + 1].Add(write); // write at sl + 1.
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
        var logger = solver.MakeSearchLog(1000, totalCyclesVar);
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

        // can't schedule buffers.
        // foreach (var (_, nodeBufferBoxs) in levelNodeBufferBoxs)
        // {
        //     foreach (var (_, bufferBoxs) in nodeBufferBoxs)
        //     {
        //         foreach (var (_, box) in bufferBoxs)
        //         {
        //             searchAbleVars.Add(box[2].Var());
        //             collector.Add(box[2].Var());
        //         }
        //     }
        // }
        var decisionBuilder = solver.MakeDefaultPhase(searchAbleVars.ToArray());
        var status = solver.Solve(decisionBuilder, new SearchMonitor[] { collector, objectiveMonitor, logger, solver.MakeSolutionsLimit(20), solver.MakeTimeLimit(50000) });
        if (!status)
        {
            resultConstructor = null!;
            return false;
        }

        var sol = collector.Solution(collector.SolutionCount() - 1);

        // dump model
        // builder IR

        resultConstructor = new TreeSolverResultConstructor(sol.ObjectiveValue(), sol, argumentsInfo, solver, opNodeMemo, tileNodeMemo, tileableNodeMemo, compileOptions);
        // var bodyBuilder = TIR.T.Sequential();
        // tree.Accept(constructor, new(bodyBuilder, Array.Empty<Expr>()));

        // var parameters = argumentsInfo.Inputs.Concat(argumentsInfo.DefUseMap.Values).Concat(argumentsInfo.Outputs).Select(k => constructor.OutSideBufferMemo[k]).ToArray();
        // var arguments = argumentsInfo.Inputs.Select(k => k.Node.Grid.Reads[k.Index]).Concat(argumentsInfo.DefUseMap.Values.Select(k => TilingUtilities.GetUninitialized(k.Node.Grid.Reads[k.Index]))).ToArray();

        // var funcBuilder = TIR.T.PrimFunc("test", "cpu", parameters).Body(bodyBuilder);
        // primFunc = funcBuilder.Build();
        // wrapper = new PrimFunctionWrapper(primFunc, parameters.Length - argumentsInfo.Outputs.Count, argumentsInfo.Inputs.Concat(argumentsInfo.DefUseMap.Values).Concat(argumentsInfo.Outputs).Select(b => b.Node.Grid.GetArgument(b.Index).CheckedType).ToArray());
        // callFunc = new Call(wrapper, arguments);
        return true;
    }

    public static List<MergePoint> EnumerateMergePoint(ITreeNode tree, int level)
    {
        var collector = new TreeMergePointCollector(level);
        tree.Accept(collector, default);
        return collector.Points;
    }

    public static Call Tile(Grid grid, CompileOptions compileOptions)
    {
        var root = new ScopeNode();
        var opId = 0;
        var totalLevel = 2;
        BuildTree(grid, root, totalLevel, ref opId);
        Dump(root, "build");

        TreeSolverResultConstructor? bestResult = null;
        foreach (var subTree in EnumerateAll(root, totalLevel, new()))
        {
            if (Solve(root, totalLevel, compileOptions, out var resultConstructor))
            {
                bestResult = (bestResult?.ObjectiveValue < resultConstructor.ObjectiveValue ? bestResult : resultConstructor) ?? resultConstructor;
            }
        }

        // try merge op2 and op1 at level 1
        // Merge(tree, 2, 1, 2);
        // Dump(tree, "merge_2_1_2");

        // Merge(tree, 2, 1, 1);
        // Dump(tree, "merge_2_1_1");

        // Merge(tree, 2, 0, 2);
        // Dump(tree, "merge_2_0_2");

        // merge 1 0 1
        // Merge(tree, 1, 0, 1);
        // Dump(tree, "merge_1_0_1");
        throw new NotSupportedException("Solve Failed");
    }

    private static List<SubTree> EnumerateAll(ITreeNode tree, int totalLevel, List<MergePoint> path)
    {
        var result = new List<SubTree>() { new(tree, new(path)) };
        for (int level = totalLevel; level > 0; level--)
        {
            var points = EnumerateMergePoint(tree, level);
            var subTrees = new List<SubTree>();
            foreach (var p in points)
            {
                var cloned = tree.Root().Clone();
                if (Merge(cloned, p.Consumer, p.Producer, level))
                {
                    Dump(cloned, p.ToString());
                    subTrees.Add(new(cloned, new(path) { p }));
                }
            }

            result.AddRange(subTrees.Select(subTree => EnumerateAll(subTree.Node, level, subTree.Paths)).SelectMany(i => i));
        }

        return result;
    }

    private record SubTree(ITreeNode Node, List<MergePoint> Paths)
    {
    }
}
