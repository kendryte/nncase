// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using Google.OrTools.Sat;
using Nncase.TIR;

namespace Nncase.Schedule.Bufferize;

public sealed class SATBufferScheduler : BufferScheduler
{
    public SATBufferScheduler(MemoryLocation memoryLocation)
        : base(memoryLocation)
    {
    }

    protected override bool TryScheduleCore(IEnumerable<BufferLifetime> lifetimes, long maxMemoryPoolSize, out long memoryPoolSize)
    {
        var model = new CpModel();
        var noOverlap = model.AddNoOverlap2D();
        var boxs = new Dictionary<TIR.PhysicalBuffer, Box>(ReferenceEqualityComparer.Instance);
        var yEnds = new List<LinearExpr>();

        int bufferId = 0;
        foreach (var lifetime in lifetimes)
        {
            var xInterval = model.NewIntervalVar(model.NewConstant(lifetime.Time.Start), model.NewConstant(lifetime.Time.Size), model.NewConstant(lifetime.Time.Stop), $"{bufferId}_x");
            var memSize = lifetime.Memory.Size;
            var maxMemStart = maxMemoryPoolSize - memSize;
            if (maxMemStart < 0)
            {
                throw new ArgumentException($"Invalid buffer size");
            }

            var memStartVar = model.NewIntVar(0, maxMemStart, $"{bufferId}_y_start");
            var yInterval = model.NewFixedSizeIntervalVar(memStartVar, memSize, $"{bufferId}_y");
            yEnds.Add(yInterval.EndExpr());

            var alignment = lifetime.Buffer.Alignment;
            model.AddModuloEquality(0, memStartVar, alignment);
            noOverlap.AddRectangle(xInterval, yInterval);
            boxs.Add(lifetime.Buffer, new(xInterval, yInterval));
            bufferId++;
        }

        var memPoolSizeVar = model.NewIntVar(0, maxMemoryPoolSize, nameof(maxMemoryPoolSize));
        model.AddMaxEquality(memPoolSizeVar, yEnds);
        model.Minimize(memPoolSizeVar);

        var solver = new CpSolver();
        solver.StringParameters = $"max_time_in_seconds:{600},num_workers:{Environment.ProcessorCount}";
        CpSolverStatus solve_status = solver.Solve(model);
        if (solve_status != CpSolverStatus.Optimal && solve_status != CpSolverStatus.Feasible)
        {
            memoryPoolSize = default;
            return false;
        }

        foreach (var lifetime in lifetimes)
        {
            lifetime.Memory.Start = checked(solver.Value(boxs[lifetime.Buffer].Y.StartExpr()));
            lifetime.Memory.Stop = checked(solver.Value(boxs[lifetime.Buffer].Y.EndExpr()));
        }

        memoryPoolSize = solver.Value(memPoolSizeVar);
        return true;
    }

    private sealed record Box(IntervalVar X, IntervalVar Y);
}
