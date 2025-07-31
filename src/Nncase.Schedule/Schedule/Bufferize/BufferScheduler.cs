// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using Microsoft.Extensions.DependencyInjection;
using Nncase.TIR;
using Nncase.Utilities;

namespace Nncase.Schedule.Bufferize;

public sealed record BufferScheduleResult(IReadOnlyDictionary<TIR.PhysicalBuffer, BufferLifetime> Buffers, long MemoryPoolSize, int Alignment);

public abstract class BufferScheduler
{
    private static readonly Type[] _bufferSchedulerTypes = [
        typeof(SATBufferScheduler),
    ];

    public BufferScheduler(MemoryLocation memoryLocation)
    {
        MemoryLocation = memoryLocation;
    }

    public MemoryLocation MemoryLocation { get; }

    public static BufferScheduleResult Schedule(MemoryLocation memoryLocation, IReadOnlyDictionary<TIR.PhysicalBuffer, BufferLifetime> lifetimes)
    {
        if (memoryLocation == MemoryLocation.Data)
        {
            foreach (var schedulerType in _bufferSchedulerTypes)
            {
                var scheduler = (BufferScheduler)ActivatorUtilities.CreateInstance(CompileSessionScope.GetCurrentThrowIfNull(), schedulerType, memoryLocation);
                if (scheduler.TrySchedule(lifetimes, out var result))
                {
                    return result;
                }
            }
        }
        else if (memoryLocation is MemoryLocation.Output or MemoryLocation.Rdata or MemoryLocation.ThreadLocalRdata or MemoryLocation.BlockLocalRdata)
        {
            var scheduler = new LinearBufferScheduler(memoryLocation);
            if (scheduler.TrySchedule(lifetimes, out var result))
            {
                return result;
            }
        }

        throw new NotSupportedException("Unable to schedule buffers");
    }

    public static IReadOnlyDictionary<MemoryLocation, BufferScheduleResult> Schedule(IReadOnlyDictionary<TIR.PhysicalBuffer, BufferLifetime> lifetimes)
    {
        var result = new Dictionary<MemoryLocation, BufferScheduleResult>();
        foreach (var group in lifetimes.GroupBy(x => x.Value.Buffer.Location))
        {
            if (group.Key is MemoryLocation.Output or MemoryLocation.Data or MemoryLocation.Rdata or MemoryLocation.ThreadLocalRdata or MemoryLocation.BlockLocalRdata)
            {
                var lifetimeDict = group.ToDictionary(x => x.Key, x => x.Value, (IEqualityComparer<TIR.PhysicalBuffer>)ReferenceEqualityComparer.Instance);
                result.Add(group.Key, Schedule(group.Key, lifetimeDict));
            }
        }

        return result;
    }

    public bool TrySchedule(IReadOnlyDictionary<TIR.PhysicalBuffer, BufferLifetime> lifetimes, [MaybeNullWhen(false)] out BufferScheduleResult result)
    {
        long maxMemoryPoolSize = 0;
        int maxAlignment = 8;
        foreach (var lifetime in lifetimes.Values)
        {
            if (lifetime.Buffer.Location != MemoryLocation)
            {
                throw new ArgumentException($"Memory location to schedule of {lifetime.Buffer} is not expected.");
            }

            var alignment = Math.Max(8, lifetime.Buffer.Alignment);
            maxMemoryPoolSize = MathUtility.AlignUp(maxMemoryPoolSize, alignment) + lifetime.Memory.Size;
            maxAlignment = Math.Max(maxAlignment, alignment);
        }

        if (TryScheduleCore(lifetimes.Values, maxMemoryPoolSize, out var memoryPoolSize))
        {
            result = new(lifetimes, memoryPoolSize, maxAlignment);
            return true;
        }
        else
        {
            result = null;
            return false;
        }
    }

    protected abstract bool TryScheduleCore(IEnumerable<BufferLifetime> lifetimes, long maxMemoryPoolSize, out long memoryPoolSize);
}
