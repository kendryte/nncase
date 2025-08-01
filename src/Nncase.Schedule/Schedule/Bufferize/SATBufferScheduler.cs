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

    protected override bool TryScheduleCore(IEnumerable<BufferLifetime> lifetimes, long maxMemoryPoolEnd, BufferScheduleOptions options, out long memoryPoolEnd)
    {
        var model = new CpModel();
        var noOverlap = model.AddNoOverlap2D();
        var boxs = new Dictionary<TIR.PhysicalBuffer, Box>(ReferenceEqualityComparer.Instance);
        var yEnds = new List<LinearExpr>();

        int bufferId = 0;
        foreach (var lifetime in lifetimes)
        {
            if (lifetime.Memory.Size == 0)
            {
                continue; // Skip buffers with zero size
            }

            var xInterval = model.NewIntervalVar(model.NewConstant(lifetime.Time.Start), model.NewConstant(lifetime.Time.Size), model.NewConstant(lifetime.Time.Stop), $"{bufferId}_x");
            var memSize = lifetime.Memory.Size;
            var maxMemStart = maxMemoryPoolEnd - memSize;
            if (maxMemStart < 0)
            {
                throw new ArgumentException($"Invalid buffer size");
            }

            var memStartVar = model.NewIntVar(options.StartAddress, maxMemStart, $"{bufferId}_y_start");
            var yInterval = model.NewFixedSizeIntervalVar(memStartVar, memSize, $"{bufferId}_y");
            yEnds.Add(yInterval.EndExpr());

            var alignment = lifetime.Buffer.Alignment;
            model.AddModuloEquality(0, memStartVar, alignment);
            noOverlap.AddRectangle(xInterval, yInterval);
            boxs.Add(lifetime.Buffer, new(xInterval, yInterval));
            bufferId++;
        }

        var memPoolEndVar = model.NewIntVar(0, maxMemoryPoolEnd, nameof(maxMemoryPoolEnd));
        model.AddMaxEquality(memPoolEndVar, yEnds);
        model.Minimize(memPoolEndVar);

        var solver = new CpSolver();
        solver.StringParameters = $"max_time_in_seconds:{600},num_workers:{Environment.ProcessorCount}";
        CpSolverStatus solve_status = solver.Solve(model);
        if (solve_status != CpSolverStatus.Optimal && solve_status != CpSolverStatus.Feasible)
        {
            memoryPoolEnd = default;
            return false;
        }

        foreach (var lifetime in lifetimes)
        {
            if (lifetime.Memory.Size == 0)
            {
                lifetime.Memory.Start = options.StartAddress;
                lifetime.Memory.Stop = options.StartAddress;
            }
            else
            {
                lifetime.Memory.Start = checked(solver.Value(boxs[lifetime.Buffer].Y.StartExpr()));
                lifetime.Memory.Stop = checked(solver.Value(boxs[lifetime.Buffer].Y.EndExpr()));
            }
        }

        memoryPoolEnd = solver.Value(memPoolEndVar);
        return true;
    }

    private sealed record Box(IntervalVar X, IntervalVar Y);
}
