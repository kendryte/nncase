// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using Google.OrTools.Sat;
using Nncase.IR;
using Nncase.Passes;
using Nncase.Passes.BufferSchedule;
using Nncase.Tests.TestFixture;
using Xunit;

namespace Nncase.Tests.BufferScheduleTest;

[AutoSetupTestMethod(InitSession = true)]
public sealed class BufferScheduleTest : TestClassBase
{
    public static TheoryData<Func<Expr>, Func<LifeTimeUpdater>, Func<BufferSizeCalculator>, int, int> TestScheduleGetItemDatas
    { get; } = new()
    {
        { SampleSwish, () => new LifeTimeUpdater(), () => new BufferSizeCalculator(), 800, 0 },
        { SampleDistSwish, () => new LifeTimeUpdater(), () => new CpuBufferSizeCalculator(), 1200, 1 },
    };

    public static Expr SampleSwish()
    {
        var a = new Var("a", new TensorType(DataTypes.Float32, new[] { 100 }));
        var b = new Var("b", new TensorType(DataTypes.Float32, new[] { 100 }));
        var c = a + b;
        var d = -c;
        var tp = new IR.Tuple(c, d);
        return IR.F.Tensors.GetItem(tp, 0) + IR.F.Tensors.GetItem(tp, 1);
    }

    public static Expr SampleDistSwish()
    {
        var ttype = new TensorType(DataTypes.Float32, new[] { 100 });
        var dtype = new DistributedType(ttype, new[] { SBP.B }, new(new[] { 1 }, "b"));
        var a = new Var("a", ttype);
        var b = new Var("b", ttype);
        var c = IR.F.CPU.Boxing(a, dtype) + IR.F.CPU.Boxing(b, dtype);
        var d = -c;
        var tp = new IR.Tuple(c, d);
        var tp0 = IR.F.Tensors.GetItem(tp, 0);
        var tp1 = IR.F.Tensors.GetItem(tp, 1);
        var e = tp0 + tp1;
        return new IR.Tuple(IR.F.CPU.Boxing(e, ttype), IR.F.CPU.Boxing(d, ttype));
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
    [MemberData(nameof(TestScheduleGetItemDatas))]
    public void TestScheduleGetItem(Func<Expr> funcGetter, Func<LifeTimeUpdater> updaterGetter, Func<BufferSizeCalculator> calcGetter, int capacity, int number)
    {
        var body = funcGetter();
#if DEBUG
        Dumpper.DumpIR(body, $"{number}_body");
#endif
        var updater = updaterGetter();
        var calc = calcGetter();
        var collector = new LifeTimeCollector(updater, calc);
        var buffers = collector.Collect(body);
        Assert.Empty(buffers.Keys.OfType<Const>());
        Assert.Empty(buffers.Keys.OfType<Var>());

        var scheduler = new BufferScheduler(capacity);
#if DEBUG
        scheduler.Dump($"{number}_start", buffers);
#endif

        scheduler.Schedule(buffers);
#if DEBUG
        scheduler.Dump($"{number}_end", buffers);
#endif
    }
}
