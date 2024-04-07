// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Google.OrTools.Sat;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.TIR;

namespace Nncase.Passes.BufferSchedule;

internal static class SchedulerSolver
{
    public static bool ScheduleByCpModel(
        IReadOnlyDictionary<Expr, ScheduledBuffer> lifenessMap,
        bool multiWorkers,
        float timeout,
        out Dictionary<Expr, ScheduledBuffer> scheduledBuffer)
    {
        scheduledBuffer = new(ReferenceEqualityComparer.Instance);
        bool invalidDomain = false;
        var model = new CpModel();

        var yMap = new Dictionary<TIR.Buffer, (IntervalVar, IntVar)>(ReferenceEqualityComparer.Instance);

        // 1. add lifeness overlap constraint
        var lifenessNoOverlap = model.AddNoOverlap2D();
        var interval_vars = lifenessMap.Where(sched => sched.Value.Buffer.MemSpan.Location == MemoryLocation.L2Data).Select(sched =>
        {
            var lifeness = lifenessMap[sched.Key].Lifeness;
            var buffer = sched.Value.Buffer.MemSpan;
            var x = model.NewIntervalVar(
                model.NewConstant(lifeness.Start),
                model.NewConstant(lifeness.End - lifeness.Start),
                model.NewConstant(lifeness.End),
                "x");

            var y_start_domain = SRAM.SramSizePerThread - ((TensorConst)buffer.Size).Value.ToScalar<int>();
            if (y_start_domain <= 0)
            {
                invalidDomain = true;
            }

            var y_start = model.NewIntVar(0, y_start_domain, $"{sched.Value.Buffer.Name}_y_start");

            var y = model.NewFixedSizeIntervalVar(
                y_start,
                ((TensorConst)buffer.Size).Value.ToScalar<long>(),
                "y");

            yMap.Add(sched.Value.Buffer, (y, y_start));

            lifenessNoOverlap.AddRectangle(x, y);
            return (x, y);
        }).ToList();

        if (invalidDomain)
        {
            return false;
        }

        var solver = new CpSolver();
        var workers = multiWorkers ? '0' : '1';
        solver.StringParameters = $"max_time_in_seconds:{timeout},num_workers:{workers}";

        var callback = new EarlyStopCallback(3);
        CpSolverStatus solve_status = solver.Solve(model, callback);

        if (solve_status == CpSolverStatus.Unknown)
        {
            return false;
        }

        if (solve_status == CpSolverStatus.ModelInvalid)
        {
            throw new InvalidDataException(model.Validate());
        }

        if (solve_status != CpSolverStatus.Optimal && solve_status != CpSolverStatus.Feasible)
        {
            return false;
        }

        foreach (var (expr, vars) in lifenessMap.Where(sched => sched.Value.Buffer.MemSpan.Location == MemoryLocation.L2Data).Select(kv => kv.Key).Zip(interval_vars))
        {
            var buffer = lifenessMap[expr].Buffer;
            var start = TIR.F.CPU.SramPtr(solver.Value(vars.y.StartExpr()), buffer.ElemType);
            var schedBuffer = buffer.With(memSpan: buffer.MemSpan.With(start: start));
            scheduledBuffer.Add(expr, new ScheduledBuffer(lifenessMap[expr].Lifeness, schedBuffer));
        }

        return true;
    }
}

internal sealed class EarlyStopCallback : CpSolverSolutionCallback
{
    private readonly int _solutionLimit;

    private int _solutionCount;

    public EarlyStopCallback(int limit)
    {
        _solutionLimit = limit;
    }

    public override void OnSolutionCallback()
    {
        _solutionCount++;
        if (_solutionCount > _solutionLimit)
        {
            StopSearch();
        }
    }
}
