// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Google.OrTools.ConstraintSolver;
using Nncase.IR;
using Nncase.Passes;
using Nncase.Schedule.TileTree;
using QuikGraph.Graphviz;
using Xunit;

namespace Nncase.Tests.AffineTest;

[TestFixture.AutoSetupTestMethod(InitSession = true)]
public sealed class UnitTestModeling : TestClassBase
{
    public UnitTestModeling()
    {
        CompileOptions.TargetOptions = new Nncase.Targets.CpuTargetOptions();
#if DEBUG
        CompileOptions.DumpFlags = Diagnostics.DumpFlags.PassIR;
#endif
    }

    [Fact]
    public void TestTwoLevelCpuCacheModeling()
    {
        var model = new Solver("tiling");
        int totlevel = 2;
        var dimValues = new int[3] { 384, 8192, 512 };
        var dimNames = new char[3] { 'm', 'n', 'k' };

        // k1,n1,m1,k2,n2,m2
        var loopNames = Enumerable.Range(0, totlevel).Select(i => dimNames.Reverse().Select(c => $"{c}_{i + 1}")).SelectMany(i => i).ToArray();
        var tileVars = loopNames.Select((n, i) => model.MakeIntVar(1, dimValues[2 - (i % dimValues.Length)], $"T_{n}")).ToArray();
        var constraints = new List<Constraint>();
        var allVars = new List<IntVar>();
        allVars.AddRange(tileVars);
        var c = model.MakeEquality(tileVars[0] * tileVars[3], dimValues[2]);
        var check = false;
        model.Add(c);
        constraints.Add(c);
        check = model.CheckConstraint(c);
        c = model.MakeEquality(tileVars[1] * tileVars[4], dimValues[1]);
        model.Add(c);
        constraints.Add(c);
        check = model.CheckConstraint(c);
        c = model.MakeEquality(tileVars[2] * tileVars[5], dimValues[0]);
        model.Add(c);
        check = model.CheckConstraint(c);

        var cacheCapacitys = new int[] { 65536, 4194304 };
        var bandwidth = new int[] { 128, 16 }; /* bytes/cycle */

        var tensors = new VTensor[3] {
            new VTensor("A", new[] { 'm', 'k' }),
            new VTensor("B", new[] { 'k', 'n' }),
            new VTensor("C", new[] { 'm', 'n' }),
        };

        // 1. paper method
        // compute dm and df
        var dm = new IntExpr[tensors.Length, loopNames.Length + 1, totlevel];
        var df = new IntExpr[tensors.Length, loopNames.Length + 1];
        for (int l = 0; l < totlevel; l++)
        {
            for (int i = 0; i < loopNames.Length + 1; i++)
            {
                if (i == 0)
                { // statement level
                    for (int a = 0; a < tensors.Length; a++)
                    {
                        if (l == 0)
                        {
                            df[a, i] = model.MakeIntConst(1 * sizeof(float));
                        }

                        dm[a, i, l] = df[a, i];
                    }
                }
                else
                {
                    var loop = loopNames[i - 1];
                    for (int a = 0; a < tensors.Length; a++)
                    {
                        if (tensors[a].ContainsLoop(loop))
                        {
                            if (l == 0)
                            {
                                df[a, i] = df[a, i - 1] * tileVars[i - 1];
                            }

                            dm[a, i, l] = dm[a, i - 1, l] * tileVars[i - 1];
                        }
                        else
                        {
                            if (l == 0)
                            {
                                df[a, i] = df[a, i - 1];
                            }

                            var v = model.MakeIntVar(0, long.MaxValue, $"dm_{a}_{i}_{l}");
                            var cc = model.MakeIfThenElseCt(
                                (df[0, i - 1] + df[1, i - 1] + df[2, i - 1]) < cacheCapacitys[l],
                                dm[a, i - 1, l],
                                dm[a, i - 1, l] * tileVars[i - 1],
                                v);
                            model.Add(cc);
                            constraints.Add(cc);
                            check = model.CheckConstraint(cc);
                            dm[a, i, l] = v;
                        }
                    }
                }
            }
        }

        // create hierarchy split branchs
        var c_dm = new IntExpr[totlevel];
        var memorys = new IntVar[loopNames.Length - 1][];

        IntExpr ComputeHierarchyDataMove(int start_i, int end_i, int l, IntExpr? original, IntExpr gateVar)
        {
            IntExpr? res = null;
            for (int a = 0; a < tensors.Length; a++)
            {
                for (int i = start_i; i <= end_i; i++)
                {
                    res = res == null ? dm[a, i, l] : res + dm[a, i, l];
                }
            }

            res = gateVar * res!;
            return original is null ? res : original + res;
        }

        for (int i1 = 1, i = 0; i1 < loopNames.Length; i1++, i++)
        {
            memorys[i] = new IntVar[loopNames.Length - i1];
            for (int i2 = i1 + 1, j = 0; i2 < loopNames.Length + 1; i2++, j++)
            {
                memorys[i][j] = model.MakeBoolVar($"l1_{i1}_l2_{i2}");
                allVars.Add(memorys[i][j]);
                c_dm[0] = ComputeHierarchyDataMove(0, i1, 0, c_dm[0], memorys[i][j]);
                c_dm[1] = ComputeHierarchyDataMove(i2, loopNames.Length, 1, c_dm[1], memorys[i][j]);
            }
        }

        // only one split strategy.
        c = model.MakeSumEquality(memorys.SelectMany(i => i).ToArray(), 1);
        model.Add(c);
        check = model.CheckConstraint(c);

        // compute time.
        // 这里如果除完不乘系数就无法开始搜索
        var totalTime = model.MakeMax((c_dm[0] + bandwidth[0] - 1) / bandwidth[0], (c_dm[1] + bandwidth[1] - 1) / bandwidth[1]);

        var decisionBuilder = model.MakePhase(allVars.ToArray(), Solver.CHOOSE_LOWEST_MIN, Solver.ASSIGN_MIN_VALUE);
        var solutionCollector = model.MakeLastSolutionCollector();
        solutionCollector.Add(allVars.ToArray());
        var objective = totalTime.Var();
        objective.SetRange(1, dimValues[0] * dimValues[1] * dimValues[2] / bandwidth[0]);
        solutionCollector.AddObjective(objective);

        var objeciveMonitor = model.MakeMinimize(objective, 1);
        var searchLog = model.MakeSearchLog(1000, objeciveMonitor);
        var searchLimit = model.MakeImprovementLimit(objective, false, 1, 0, 1, 2);
        var timeLimit = model.MakeTimeLimit(50000);

        // var assign = new Assignment(model);
        // assign.Add(allVars.ToArray());
        // assign.SetValue(tileVars[0], 1);
        // assign.SetValue(tileVars[1], 1);
        // assign.SetValue(tileVars[2], 1);
        // assign.SetValue(tileVars[3], dimValues[2]); // k2
        // assign.SetValue(tileVars[4], dimValues[1]); // n2
        // assign.SetValue(tileVars[5], dimValues[0]); // m2
        // for (int i1 = 1, i = 0; i1 < loopNames.Length; i1++, i++)
        // {
        //     for (int i2 = i1 + 1, j = 0; i2 < loopNames.Length + 1; i2++, j++)
        //     {
        //         if (i == 3 && j == 0)
        //         {
        //             assign.SetValue(memorys[i][j], 1);
        //         }
        //         else
        //         {
        //             assign.SetValue(memorys[i][j], 0);
        //         }
        //     }
        // }
        // if (!model.CheckAssignment(assign))
        // {
        //     for (int i = 0; i < constraints.Count; i++)
        //     {
        //         var x = model.CheckConstraint(constraints[i]);
        //         System.Console.WriteLine(constraints[i].Name());
        //     }
        //     System.Console.WriteLine("fuck");
        // }
        model.Solve(decisionBuilder, new SearchMonitor[] { objeciveMonitor, timeLimit, searchLog, searchLimit, solutionCollector });
        if (solutionCollector.SolutionCount() < 1)
        {
            throw new InvalidOperationException();
        }

        var solution = solutionCollector.SolutionCount() - 1;

#if DEBUG
        for (int i = 0; i < tileVars.Length; i++)
        {
            System.Console.WriteLine(tileVars[i].Name() + " : " + solutionCollector.Value(solution, tileVars[i]));
        }

        var splitPoint = new long[loopNames.Length - 1][];
        for (int i1 = 1, i = 0; i1 < loopNames.Length; i1++, i++)
        {
            splitPoint[i] = new long[loopNames.Length - i1];
            for (int i2 = i1 + 1, j = 0; i2 < loopNames.Length + 1; i2++, j++)
            {
                splitPoint[i][j] = solutionCollector.Value(solution, memorys[i][j]);
                System.Console.WriteLine(memorys[i][j].Name() + " : " + splitPoint[i][j]);
            }
        }
#endif
    }

    [Fact]
    public void TestTwoLevelMemoryModeling()
    {
        Nncase.Schedule.MatmulTiler.Solve();
    }

    [Fact]
    public void TestMultiInputs()
    {
        var func = FunctionSamples.Get4();

        var post = CompilerServices.Rewrite(func, new IRewriteRule[] { new Passes.Rules.CPU.Affine.LowerUnary(), new Passes.Rules.CPU.Affine.LowerBinary(), }, new());
#if DEBUG
        Dumpper.DumpIR(post, "post");
#endif

        var grid = (IR.Affine.Grid)((IR.Function)post).Body;

        var root = TreeBuilder.Build(grid, 2);
#if DEBUG
        root.Dump("build");
#endif

        Assert.Equal(4, root.Collect().OfType<OpNode>().Count());

        Assert.True(root.Merge(2, 1, 2));
        Assert.True(root.Merge(2, 0, 2));
#if DEBUG
        root.Dump("merged");
#endif

        Assert.IsType<TreeSolverResultConstructor>(Schedule.TreeTiler.Solve(root, CompileOptions.TargetOptions));
    }

    [Fact]
    public void TestMultiInputs2()
    {
        var func = FunctionSamples.Get1();

        var post = CompilerServices.ERewrite(func, new IRewriteRule[] { new Passes.Rules.CPU.PackMatMul(1, 4), new Passes.Rules.CPU.PackUnary(1, 4) }, new(), CompileOptions);
        post = CompilerServices.Rewrite(post, new IRewriteRule[] { new Passes.Rules.CPU.Affine.LowerPack(), new Passes.Rules.CPU.Affine.LowerUnary(), new Passes.Rules.CPU.Affine.LowerMatmul(), }, new());
#if DEBUG
        Dumpper.DumpIR(post, "post");
#endif

        var call = (IR.Call)((IR.Function)post).Body;
        var grid = (IR.Affine.Grid)call.Arguments[0];

        var root = TreeBuilder.Build(grid, 2);
#if DEBUG
        root.Dump("build");
#endif

        Assert.Equal(7, root.Collect().OfType<OpNode>().Count());

        Assert.True(root.Merge(2, 1, 2));
        Assert.True(root.Merge(2, 0, 2));
        Assert.True(root.Merge(6, 5, 2));
        Assert.True(root.Merge(6, 4, 2));
#if DEBUG
        root.Dump("merged");
#endif

        Assert.IsType<TreeSolverResultConstructor>(Schedule.TreeTiler.Solve(root, CompileOptions.TargetOptions));
    }

    [Fact]
    public void TestTreeTiler()
    {
        Function func;
        {
            var a = new Var(new TensorType(DataTypes.Float32, new[] { 128, 256 }));
            var b = new Var(new TensorType(DataTypes.Float32, new[] { 256, 384 }));
            var c = IR.F.Tensors.MatMul(a, b);
            var d = IR.F.Math.Exp(c);
            var e = new Var(new TensorType(DataTypes.Float32, new[] { 384, 512 }));
            var f = IR.F.Tensors.MatMul(d, e);
            func = new(f, a, b, e);
        }

        var post = CompilerServices.Rewrite(func, new IRewriteRule[] { new Passes.Rules.CPU.Affine.LowerUnary(), new Passes.Rules.CPU.Affine.LowerMatmul(), }, new());
        Dumpper.DumpIR(post, "post");

        if (post is not Function { Body: IR.Affine.Grid grid })
        {
            return;
        }

        Schedule.TreeTiler.Tile(grid, Nncase.Targets.CPUTarget.Kind, 0, CompileOptions.TargetOptions);
    }

    [Fact]
    public void TestTilePackMatmul()
    {
        Function func;
        {
            var a = new Var(new TensorType(DataTypes.Float32, new[] { 1024, 2048 }));
            var b = new Var(new TensorType(DataTypes.Float32, new[] { 2048, 1024 }));
            var c = IR.F.Tensors.MatMul(a, b);
            var d = IR.F.Math.Exp(c);
            var e = new Var(new TensorType(DataTypes.Float32, new[] { 1024, 3072 }));
            var f = IR.F.Tensors.MatMul(d, e);
            func = new(f, a, b, e);
        }

        var post = CompilerServices.ERewrite(func, new IRewriteRule[] { new Passes.Rules.CPU.PackMatMul(1, 8), new Passes.Rules.CPU.PackUnary(1, 8), }, new(), CompileOptions);
        Dumpper.DumpIR(post, "pack");
        post = CompilerServices.Rewrite(post, new IRewriteRule[] { new Passes.Rules.CPU.Affine.LowerPack(), new Passes.Rules.CPU.Affine.LowerUnary(), new Passes.Rules.CPU.Affine.LowerMatmul(), }, new());
        Dumpper.DumpIR(post, "grid");

        // if (post is not Function { Body: IR.Affine.Grid grid })
        // {
        //     return;
        // }

        // Schedule.TreeTiler.Tile(grid, Nncase.Targets.CPUTarget.Kind, 0, CompileOptions.TargetOptions);
    }

    [Fact]
    public void TestAutoFusion()
    {
        var func = FunctionSamples.Get1();
        var module = new IR.IRModule(func);
        CompileSession.Compiler.ImportIRModule(module);
        CompileSession.Compiler.CompileAsync();
        using (var stream = Diagnostics.DumpScope.Current.OpenFile("test.kmodel"))
        {
            CompileSession.Compiler.Gencode(stream);
        }
    }

    [Fact]
    public void TestAutoFusionFailedCase0()
    {
        var func = FunctionSamples.Get3();
        var post = CompilerServices.Rewrite(func, new IRewriteRule[] { new Passes.Rules.CPU.Affine.LowerBinary() }, new());
        Dumpper.DumpIR(post, "post");

        if (post is not Function { Body: IR.Affine.Grid grid })
        {
            return;
        }

        var root = new ScopeNode();
        var opId = 0;
        var totalLevel = 2;
        Schedule.TreeTiler.BuildTree(grid, root, totalLevel, ref opId);
#if DEBUG
        root.Dump("built");
#endif
        root.Merge(3, 2, 2);
        root.Merge(3, 1, 2);
        root.Merge(3, 0, 2);
        root.Merge(3, 2, 1);
        root.Merge(3, 1, 1);
        root.Merge(3, 0, 1);
#if DEBUG
        root.Dump("fused");
#endif

        var result = Schedule.TreeTiler.Solve(root, CompileOptions.TargetOptions);

        if (result is not null)
        {
            result.ConstructResult("cpu", 0);
        }
    }

    [Fact]
    public void TestMerge()
    {
        var func = FunctionSamples.Get1();
        var post = CompilerServices.Rewrite(func, new IRewriteRule[] { new Passes.Rules.CPU.Affine.LowerUnary(), new Passes.Rules.CPU.Affine.LowerMatmul(), }, new());
        Dumpper.DumpIR(post, "post");

        if (post is not Function { Body: IR.Affine.Grid grid })
        {
            return;
        }

        // Schedule.TreeTiler.BuildTree(grid, CompileOptions.TargetOptions);
        var root = new Schedule.TileTree.ScopeNode();
        var opId = 0;
        var totalLevel = 2;
        Schedule.TreeTiler.BuildTree(grid, root, totalLevel, ref opId);
        root.Dump("build");

        root.Merge(2, 1, 2);
        var m1 = root.Clone();
        m1.Dump("0");
        m1.Merge(2, 0, 2);
        var m2 = m1.Clone();
        m2.Dump("1");
        m2.Merge(1, 0, 1);
        var m3 = m2.Clone();
        m3.Dump("2");
        m3.Merge(2, 1, 1);
        var m4 = m3.Clone();
        m4.Dump("3");
    }

    [Fact]
    public void TestGetArgumentInfo()
    {
        var func = FunctionSamples.Get1();
        var post = CompilerServices.Rewrite(func, new IRewriteRule[] { new Passes.Rules.CPU.Affine.LowerUnary(), new Passes.Rules.CPU.Affine.LowerMatmul(), }, new());
        Dumpper.DumpIR(post, "post");

        if (post is not Function { Body: IR.Affine.Grid grid })
        {
            return;
        }

        var root = new ScopeNode();
        var opId = 0;
        var totalLevel = CompileOptions.TargetOptions.MemoryCapacities.Length - 1;
        Schedule.TreeTiler.BuildTree(grid, root, totalLevel, ref opId);
        root.Dump("build");
        var m1 = root.Root<ITreeNode>().Clone();
        m1.Merge(2, 1, 2);
        m1.Dump("final");

        var res = TreeSolverInitializer.Init(m1, totalLevel, CompileOptions.TargetOptions, out _, out _, out _, out _);
        Assert.Equal(3, res.Inputs.Count);
        Assert.Single(res.Outputs);
        Assert.Equal(2, res.DefUseMap.Keys.Count);
    }

    [Fact]
    public void TestGetArgumentInfo2()
    {
        var func = FunctionSamples.Get2();
        var post = CompilerServices.Rewrite(func, new IRewriteRule[] { new Passes.Rules.CPU.Affine.LowerUnary(), new Passes.Rules.CPU.Affine.LowerMatmul(), new Passes.Rules.CPU.Affine.LowerBinary() }, new());
        Dumpper.DumpIR(post, "post");

        if (post is not Function { Body: IR.Affine.Grid grid })
        {
            return;
        }

        var root = new ScopeNode();
        var opId = 0;
        var totalLevel = CompileOptions.TargetOptions.MemoryCapacities.Length - 1;
        Schedule.TreeTiler.BuildTree(grid, root, totalLevel, ref opId);
        root.Dump("build");
        var m1 = root.Root<ITreeNode>().Clone();
        m1.Merge(1, 0, 2);
        m1.Merge(1, 0, 1);
        m1.Dump("final");

        var res = TreeSolverInitializer.Init(m1, totalLevel, CompileOptions.TargetOptions, out _, out _, out _, out _);
        Assert.Equal(4, res.Inputs.Count);
        Assert.Single(res.Outputs);
        Assert.Single(res.DefUseMap.Keys);
    }

    [Fact]
    public void TestGetArgumentInfo3()
    {
        var func = FunctionSamples.Get2();
        var post = CompilerServices.Rewrite(func, new IRewriteRule[] { new Passes.Rules.CPU.Affine.LowerUnary(), new Passes.Rules.CPU.Affine.LowerMatmul(), new Passes.Rules.CPU.Affine.LowerBinary() }, new());
        Dumpper.DumpIR(post, "post");

        if (post is not Function { Body: IR.Affine.Grid grid })
        {
            return;
        }

        var root = new ScopeNode();
        var opId = 0;
        var totalLevel = CompileOptions.TargetOptions.MemoryCapacities.Length - 1;
        Schedule.TreeTiler.BuildTree(grid, root, totalLevel, ref opId);
        root.Dump("build");
        var m1 = root.Root<ITreeNode>().Clone();
        m1.Merge(1, 0, 2);
        m1.Dump("final");

        var res = TreeSolverInitializer.Init(m1, totalLevel, CompileOptions.TargetOptions, out _, out _, out _, out _);

        // when merge point at top level, should put the cache buffer into defuse map.
        Assert.Equal(4, res.Inputs.Count);
        Assert.Single(res.Outputs);
        Assert.Equal(2, res.DefUseMap.Keys.Count);
    }

    [Fact]
    public void TestSolveNoOverlapping()
    {
        var solver = new Solver("a");
        var asize = solver.MakeIntConst(1);
        var csize = solver.MakeIntConst(2);
        var aplace = solver.MakeBoolVar();
        var cplace = solver.MakeBoolVar();
        solver.Add(solver.MakeSumEquality(new[] { aplace, cplace }, 1));

        var offset = solver.MakeIntVar(0, 2);

        var aLife = (solver.MakeIntConst(0), solver.MakeIntConst(1));
        var aSpan = (offset, aplace * asize);
        var cLife = (solver.MakeIntConst(0), solver.MakeIntConst(3));
        var cSpan = (offset, cplace * csize);
        var cons = solver.MakeNonOverlappingBoxesConstraint(new[] { aLife.Item1, cLife.Item1 }, new[] { aSpan.offset, cSpan.offset }, new[] { aLife.Item2, cLife.Item2 }, new[] { aSpan.Item2.Var(), cSpan.Item2.Var() });
        solver.Add(cons);

        var decisionBuilder = solver.MakeDefaultPhase(new[] { offset, aplace, cplace });
        var collector = solver.MakeLastSolutionCollector();
        collector.Add(new[] { offset, aplace, cplace });
        var status = solver.Solve(decisionBuilder, new SearchMonitor[] { collector, solver.MakeSolutionsLimit(20) });

        // note verified the [0,1] and [0,0] is not overlapping.
        Assert.True(status);

        var sol = collector.Solution(0);
        System.Console.WriteLine(sol.Value(offset));
        System.Console.WriteLine(sol.Value(aplace));
        System.Console.WriteLine(sol.Value(cplace));
    }
}

internal sealed record VTensor(string Name, char[] Dims)
{
    public bool ContainsLoop(string loopName)
    {
        foreach (var dim in Dims)
        {
            if (loopName.Contains(dim, System.StringComparison.Ordinal))
            {
                return true;
            }
        }

        return false;
    }
}
