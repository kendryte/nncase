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
        var vars = Enumerable.Range(0, domainDims).Select(i => $"op{copId}_d{i}").ToArray();
        if (current.Body[0] is not Call { Target: Op op })
        {
            throw new InvalidOperationException("body is not call");
        }

        var opNode = new OpNode(current, op, current.Buffers.ToArray(), copId, vars, domain, bufferShapes, current.AccessMaps.ToArray(), dependences.ToArray());
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
            var printer = new TreePrinter(writer);
            tree.Accept(printer, TreePrinterContext.Default);
            writer.Flush();
        }
    }

    public static void Merge(ITreeNode tree, int opConsumer, int opProducer, int level)
    {
        var merger = new TreeMerger(opConsumer, opProducer, level);
        tree.Accept(merger, default);
    }

    public static bool Solve(ITreeNode tree, int totalLevel, CompileOptions compileOptions, out Call callFunc, out PrimFunctionWrapper wrapper, out TIR.PrimFunction primFunc)
    {
        long[] memoryCapacitys = new long[] { 2 * 1024 * 1024, int.MaxValue }; // l1, l2
        long[] memoryBandWidths = new long[] { 256, 128, 4 }; // l0, l1, l2
        var solver = new Solver("treeSolver");
        var one = solver.MakeIntConst(1);
        var zero = solver.MakeIntConst(0);
        var elem = solver.MakeIntConst(4);
        var opNodeMemo = new Dictionary<OpNode, OpNodeInfo>();
        var tileNodeMemo = new Dictionary<TileNode, TileNodeInfo>();
        var tileableNodeMemo = new Dictionary<ITileAbleNode, DomainInfo>();
        var init = new TreeSolverInitializer(totalLevel, solver, one, zero, elem, opNodeMemo, tileNodeMemo, tileableNodeMemo, compileOptions.TargetOptions);
        tree.Accept(init, TreeSolverInitializer.Context.Default);
        var initWrites = new TreeSolverWritesInitializer(solver, one, zero, elem, opNodeMemo, tileNodeMemo, tileableNodeMemo, compileOptions.TargetOptions);
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
                        var parentStored = solver.MakeIsEqualVar(solver.MakeSum(parentNodeInfo.BufferInfoMap[pbid].Places.Select(p => p[l - 1]).ToArray()), one);
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
        var levelNodeBufferStart = new Dictionary<int, IReadOnlyCollection<IntExpr>>();
        var levelNodeBufferDuration = new Dictionary<int, IReadOnlyCollection<IntExpr>>();
        var levelNodeBufferOffset = new Dictionary<int, IReadOnlyCollection<IntVar>>();
        var levelNodeBufferExtent = new Dictionary<int, IReadOnlyCollection<IntExpr>>();
        for (int l = 1; l < totalLevel; l++)
        {
            var bufferStarts = new List<IntExpr>();
            var bufferDurations = new List<IntExpr>();
            var bufferOffsets = new List<IntVar>();
            var bufferExtents = new List<IntExpr>();
            foreach (var (node, nodeInfo) in tileNodeMemo.Where(p => p.Key.Level >= l))
            {
                foreach (var (bid, bufferInfo) in nodeInfo.BufferInfoMap)
                {
                    bufferStarts.Add(solver.MakeIntConst(bufferInfo.Lifeness.Item1, $"start[{l}, op{node.OpId}, b{bid.Index}]"));
                    bufferDurations.Add(solver.MakeIntConst(bufferInfo.Lifeness.Item2 - bufferInfo.Lifeness.Item1));
                    bufferOffsets.Add(solver.MakeIntVar(0, memoryCapacitys[l - 1] - 1, $"off[{l}, op{node.OpId}, b{bid.Index}]"));
                    var extexts = bufferInfo.Places.Select(p => p[l - 1]).Zip(bufferInfo.SizeVars).Select(p => p.First * p.Second).ToArray();
                    bufferExtents.Add(extexts.Skip(1).Aggregate(extexts[0], solver.MakeSum));
                }
            }

            levelNodeBufferStart[l] = bufferStarts;
            levelNodeBufferDuration[l] = bufferDurations;
            levelNodeBufferOffset[l] = bufferOffsets;
            levelNodeBufferExtent[l] = bufferExtents;
            var x_vars = bufferStarts.Select(i => i.Var()).ToArray();
            var y_vars = bufferOffsets.ToArray();
            var x_sizes = bufferDurations.Select(i => i.Var()).ToArray();
            var y_sizes = bufferExtents.Select(i => i.Var()).ToArray();
            solver.Add(solver.MakeLessOrEqual(bufferExtents.Skip(1).Aggregate(bufferExtents[0], solver.MakeSum), memoryCapacitys[l - 1]));

            // can't schedule buffers now.
            // solver.Add(solver.MakeNonOverlappingBoxesConstraint(x_vars, y_vars, x_sizes, y_sizes));
            // foreach (var topOffset in y_vars.Zip(y_sizes).Select(p => p.First + p.Second))
            // {
            //     solver.Add(solver.MakeLessOrEqual(topOffset, memoryCapacitys[l - 1]));
            // }
        }

        // compute the cycles as objective
        var levelDataReads = Enumerable.Range(0, totalLevel + 1).Select(i => (IntExpr)zero).ToArray();
        var levelDataWrites = Enumerable.Range(0, totalLevel + 1).Select(i => (IntExpr)zero).ToArray();
        IntExpr computeCycles = zero;

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
                var l1Load = loopTrip * opNodeInfo.Size.SkipLast(1).Aggregate((IntExpr)zero, (acc, s) => acc + s);
                levelDataReads[1] += l1Load; // read from level 1
                var l0Store = loopTrip * opNodeInfo.Size[^1];
                levelDataWrites[0] += l0Store; // write to level 0

                // note use kernel info. amx 32*32 matmul only 4 cycles
                computeCycles += 4 * loopTrip;
            }
        }

        for (int l = 1; l < totalLevel; l++)
        {
            var currentLevelWrites = new List<IntExpr>();
            var restParentLevelNums = totalLevel - l;
            var parentLevelReads = Enumerable.Range(0, restParentLevelNums).Select(i => new List<IntExpr>()).ToArray();
            foreach (var (childNode, childNodeInfo) in tileNodeMemo.Where(p => p.Key.Level == l))
            {
                foreach (var (childBid, childBufferInfo) in childNodeInfo.BufferInfoMap)
                {
                    // store at current level
                    var writes = childBufferInfo.Places.Select(p => p[l - 1]).Zip(childBufferInfo.Writes).Select(p => p.First * p.Second);
                    currentLevelWrites.AddRange(writes);  // write to l
                    parentLevelReads[0].AddRange(writes); // read from l + 1

                    for (int pl = l + 1; pl <= totalLevel; pl++)
                    {
                        foreach (var (parentNode, parentBufferInfo) in tileNodeMemo.Where(p => p.Key.Level == pl).Select(p => (p.Key, p.Value.BufferInfoMap[p.Value.GetCacheBid(childBid)])))
                        {
                            // parent level store at current level
                            writes = parentBufferInfo.Places.Select(p => p[l - 1]).Zip(parentBufferInfo.Writes).Select(p => p.First * p.Second);
                            currentLevelWrites.AddRange(writes); // pl write to l
                            if (pl < totalLevel)
                            {
                                parentLevelReads[pl - l].AddRange(writes); // read from pl + 1
                            }
                        }
                    }
                }
            }

            levelDataWrites[l] += currentLevelWrites.Skip(1).Aggregate(currentLevelWrites.First(), solver.MakeSum);
            for (int i = 0; i < parentLevelReads.Length; i++)
            {
                levelDataReads[l + i] += parentLevelReads[i].Skip(1).Aggregate(parentLevelReads[i].First(), solver.MakeSum);
            }
        }

        var memoryCycles = Enumerable.Range(0, totalLevel + 1).Select(i => (IntExpr)zero).ToArray();
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
        // foreach (var (_, bufferOffset) in levelNodeBufferOffset)
        // {
        //     searchAbleVars.AddRange(bufferOffset);
        //     collector.Add(bufferOffset.ToArray());
        // }
        var decisionBuilder = solver.MakeDefaultPhase(searchAbleVars.ToArray());
        var status = solver.Solve(decisionBuilder, new SearchMonitor[] { collector, objectiveMonitor, logger, solver.MakeSolutionsLimit(20), solver.MakeTimeLimit(50000) });
        if (!status)
        {
            callFunc = null!;
            wrapper = null!;
            primFunc = null!;
            return false;
        }

        var sol = collector.Solution(collector.SolutionCount() - 1);

        // dump model
        using (var stream = Diagnostics.DumpScope.Current.OpenFile($"model.py"))
        {
            using var baseWriter = new StreamWriter(stream);
            var printer = new TreeSolverPrinter(baseWriter, sol, solver, one, zero, elem, opNodeMemo, tileNodeMemo, tileableNodeMemo, compileOptions.TargetOptions);
            tree.Accept(printer, default);
            using var writer = new System.CodeDom.Compiler.IndentedTextWriter(baseWriter, "  ");
            writer.WriteLine("tileVarConstraints:");
            writer.Indent++;
            foreach (var (opnode, consts) in tileVarConstraints)
            {
                TreeSolverPrinter.WriteIntExprVector(writer, opnode.ToString(), consts, sol);
            }

            writer.Indent--;

            writer.WriteLine("lowestStoreBufferNumsConstrains:");
            writer.Indent++;
            foreach (var (node, cons) in lowestStoreBufferNumsConstrains)
            {
                TreeSolverPrinter.WriteIntExprVector(writer, node.ToString(), new[] { cons }, sol);
            }

            writer.Indent--;

            writer.WriteLine("EachParentNodeCreateBufferConstraints:");
            writer.Indent++;
            foreach (var (node, constraints) in eachParentNodeCreateBufferConstraints)
            {
                TreeSolverPrinter.WriteIntExprVector(writer, node.ToString(), constraints.Values.ToArray(), sol);
            }

            writer.Indent--;

            writer.WriteLine("memoryCapacityConstraints:");
            writer.Indent++;

            // for (int l = 1; l < totalLevel; l++)
            // {
            //     TreeSolverPrinter.WriteIntExprVector(writer, l.ToString(), levelNodeBufferOffset[l].ToArray(), sol);
            //     TreeSolverPrinter.WriteIntExprVector(writer, l.ToString(), levelNodeBufferExtent[l].ToArray(), sol);
            // }
            writer.Indent--;

            TreeSolverPrinter.WriteIntExprVector(writer, "levelDataReads:", levelDataReads, sol);
            TreeSolverPrinter.WriteIntExprVector(writer, "levelDataWrites:", levelDataWrites, sol);
            TreeSolverPrinter.WriteIntExprVector(writer, "memoryCycles:", memoryCycles, sol);
            writer.WriteLine($"computeCycles: {computeCycles.ToSimplifyString()}");
        }

        // builder IR
        var argsCollector = new TreeSolverArgumentsCollector();
        tree.Accept(argsCollector, default);
        var constructor = new TreeSolverResultConstructor(sol, argsCollector.Inputs, argsCollector.Outputs, solver, one, zero, elem, opNodeMemo, tileNodeMemo, tileableNodeMemo, compileOptions);
        var bodyBuilder = TIR.T.Sequential();
        tree.Accept(constructor, new(bodyBuilder, Array.Empty<Expr>()));
        var rootInfo = tileNodeMemo[(TileNode)tree.GetChildTileableNode()!];
        var keys = constructor.BufferMemo.Keys.ToList();
        foreach (var (bid, binfo) in rootInfo.BufferInfoMap)
        {
            if (rootInfo.DefUseMap.TryGetValue(bid, out var sourcebid))
            {
                keys.Remove(bid);
                keys.Remove(sourcebid);
            }
        }

        var parameters = argsCollector.Inputs.Select(k => constructor.BufferMemo[k]).Concat(argsCollector.Outputs.Select(k => constructor.BufferMemo[k])).ToArray();
        var arguments = argsCollector.Inputs.Select(k => k.Node.Grid.Reads[k.Index]).ToArray();

        var funcBuilder = TIR.T.PrimFunc("test", "cpu", parameters).Body(bodyBuilder);
        primFunc = funcBuilder.Build();
        wrapper = new PrimFunctionWrapper(primFunc, argsCollector.Inputs.Count, argsCollector.Inputs.Concat(argsCollector.Outputs).Select(b => b.Node.Grid.GetArgument(b.Index).CheckedType).ToArray());
        callFunc = new Call(wrapper, arguments);
        return true;
    }

    public static TileResult Tile(Grid grid, CompileOptions compileOptions)
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

        // merge 1 0 1
        // Merge(tree, 1, 0, 1);
        // Dump(tree, "merge_1_0_1");
        if (Solve(tree, maxLevel, compileOptions, out var call, out var wrapper, out var primFunc))
        {
            // Diagnostics.DumpScope.Current.DumpIR(call, "func");
            return new TileResult(call, wrapper, primFunc);
        }

        throw new NotSupportedException("fuck");
    }
}

public record TileResult(Call Call, PrimFunctionWrapper Wrapper, PrimFunction PrimFunc)
{
}
