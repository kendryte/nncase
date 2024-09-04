// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using System.Threading.Tasks;
using Google.OrTools.Sat;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.Passes;
using Nncase.Passes.BufferSchedule;
using Nncase.Tests.TestFixture;
using Xunit;

namespace Nncase.Tests.BufferScheduleTest;

[AutoSetupTestMethod(InitSession = true)]
public sealed class UnitTestBufferScheduler : TestClassBase
{
    public UnitTestBufferScheduler()
    {
        DefaultTargetName = Targets.CPUTarget.Kind;
        CompileOptions.TargetOptions = new Targets.CpuTargetOptions();
#if DEBUG
        CompileOptions.DumpFlags = Diagnostics.DumpFlags.PassIR | Diagnostics.DumpFlags.Rewrite | Diagnostics.DumpFlags.CodeGen | Diagnostics.DumpFlags.Schedule | Diagnostics.DumpFlags.EGraphCost | Diagnostics.DumpFlags.Tiling;
#endif
    }

    public static TheoryData<Func<Fusion>, int, int> ScheduleGetItemDatas
    { get; } = new()
    {
        { SampleSwish, 1600, 0 },
    };

    public static Fusion SampleSwish()
    {
        var ttype = new TensorType(DataTypes.Float32, new[] { 100 });
        var dtype = new DistributedType(ttype, new[] { SBP.B }, new(new[] { 1 }, "b"));
        var a = new Var("a", ttype);
        var b = new Var("b", ttype);
        var boxa = IR.F.CPU.Boxing(a, dtype);
        var boxb = IR.F.CPU.Boxing(b, dtype);
        var tp = new IR.Tuple([boxa, boxb]);
        var tc = new TensorConst(Tensor.FromScalar(1.0f, [100]), new[] { SBP.B }, new(new[] { 1 }, "b"));
        var c = IR.F.Math.Sin(tc);
        var d = IR.F.Math.Cos(c);
        var e = IR.F.Math.Neg(d);
        var f = IR.F.Math.Abs(e);
        var g = IR.F.Math.Cos(f);
        var h = IR.F.Math.Neg(g);
        var i = IR.F.Tensors.GetItem(tp, 1) + h;

        var body = new IR.Tuple(IR.F.CPU.Boxing(IR.F.Tensors.GetItem(tp, 0), ttype), IR.F.CPU.Boxing(i, ttype));
        return new Fusion("kernel", Targets.CPUTarget.Kind, body, a, b);
    }

    [Fact]
    public void TestNoOverLapWithZeroSize()
    {
        var memCapcity = 10;
        var model = new CpModel();
        var cons = model.AddNoOverlap2D();
        var ax = model.NewIntervalVar(0, 2, 2, "ax");
        var ay_start = model.NewIntVar(0, 0, "ay_start");
        var ay = model.NewFixedSizeIntervalVar(ay_start, memCapcity, "ay");

        var y_size = 0;
        var bx = model.NewIntervalVar(1, 2, 3, "bx");
        var by_start = model.NewIntVar(0, memCapcity - y_size, "by_start");
        var by = model.NewFixedSizeIntervalVar(by_start, y_size, "by");
        cons.AddRectangle(ax, ay);
        cons.AddRectangle(bx, by);

        var solver = new CpSolver();
        CpSolverStatus solve_status = solver.Solve(model);
        Assert.Equal(CpSolverStatus.Optimal, solve_status);
        System.Console.WriteLine(solver.Value(by_start));
    }

    [Theory]
    [MemberData(nameof(ScheduleGetItemDatas))]
    public async Task TestScheduleGetItem(Func<Fusion> fusionGetter, int capacity, int number)
    {
        ((Targets.CpuTargetOptions)CompileOptions.TargetOptions).HierarchySizes[^1] = capacity;
        var fusion = fusionGetter();
        var dupVars = fusion.Parameters.AsValueEnumerable().Select(v => new Var(v.TypeAnnotation)).ToArray();

        var module = new IRModule(new Function("main", new Call(fusion, dupVars), dupVars));
        module.Add(fusion);

        var inputs = dupVars.AsValueEnumerable().Select(v =>
        {
            var ttype = v.CheckedTensorType;
            return IR.F.Random.Normal(ttype.DType, ttype.Shape.ToValueArray()).Evaluate().AsTensor();
        }).ToArray();

        var kernelCase = new ModuleCase($"case{number}", module, dupVars, inputs);
        await Testing.CompileAndRun(kernelCase, CompileOptions, CompileSession, Compile);
    }

    private async Task Compile(IRModule module)
    {
        var passManager = CompileSession.CreatePassManager("pmgr");
        passManager.Add<CPUFusionToTirPass>();

        // todo add auto fusion merge pass here.
        passManager.Add<PrimFuncPass>().Configure(p =>
        {
            p.Add<Passes.Mutators.UnFoldBlock>();
            p.Add<Passes.Mutators.FlattenSequential>();
            p.Add<Passes.Mutators.TailLoopStripping>();
            p.Add<Passes.Mutators.FoldConstCall>();
        });

        passManager.AddWithName<DDrBufferSchdeulePass>("DDrBufferSchdeule");

        passManager.AddWithName<PrimFuncPass>("InstStage").Configure(p =>
        {
            p.Add<Passes.Mutators.FlattenBuffer>();
            p.Add<Passes.Mutators.FoldConstCall>();
            p.Add<Passes.Mutators.RemoveNop>();
        });
        CompileSession.Target.RegisterTargetDependentBeforeCodeGen(passManager, CompileSession.CompileOptions);
        await passManager.RunAsync(module);
    }
}
