// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Reactive;
using System.Runtime.InteropServices;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.TIR;

namespace Nncase.Schedule.Bufferize;

public sealed record LifetimeCollectionResult(TIR.Buffer[] Buffers, IReadOnlyDictionary<TIR.PhysicalBuffer, BufferLifetime> Lifetimes);

public sealed class LifetimeCollector
{
    public LifetimeCollectionResult Collect(Expr expr)
    {
        var buffers = new HashSet<TIR.Buffer>(ReferenceEqualityComparer.Instance);
        var lifetimes = new Dictionary<TIR.PhysicalBuffer, (BufferLifetime Lifetime, int RefCount)>(ReferenceEqualityComparer.Instance);
        new BufferCollector(buffers, lifetimes).Visit(expr);
        new LifetimeRecoder(lifetimes).Visit(expr);
        ValidateZeroRefCounts(lifetimes);
        return new(buffers.ToArray(), lifetimes.ToDictionary(x => x.Key, x => x.Value.Lifetime, (IEqualityComparer<TIR.PhysicalBuffer>)ReferenceEqualityComparer.Instance));
    }

    private static bool TryGetPhysicalBuffer(BaseExpr expr, [MaybeNullWhen(false)] out TIR.Buffer buffer, [MaybeNullWhen(false)] out TIR.PhysicalBuffer physicalBuffer)
    {
        switch (expr)
        {
            case TIR.Buffer b:
                buffer = b;
                physicalBuffer = b.MemSpan.Buffer;
                return true;
            default:
                break;
        }

        buffer = null;
        physicalBuffer = null;
        return false;
    }

    private static void ValidateZeroRefCounts(Dictionary<TIR.PhysicalBuffer, (BufferLifetime Lifetime, int RefCount)> lifetimes)
    {
        foreach (var (_, refCount) in lifetimes.Values)
        {
            if (refCount != 0)
            {
                throw new InvalidOperationException($"Non-zero ref count found");
            }
        }
    }

    private sealed class BufferCollector : ExprWalker
    {
        private readonly HashSet<TIR.Buffer> _buffers;
        private readonly Dictionary<TIR.PhysicalBuffer, (BufferLifetime Lifetime, int RefCount)> _lifetimes;

        public BufferCollector(HashSet<TIR.Buffer> buffers, Dictionary<TIR.PhysicalBuffer, (BufferLifetime Lifetime, int RefCount)> lifetimes)
        {
            _buffers = buffers;
            _lifetimes = lifetimes;
        }

        protected override Unit VisitLeafPhysicalBuffer(TIR.PhysicalBuffer expr)
        {
            if (expr.Start is None or Call { Target: IR.Buffers.AddressOf })
            {
                var bufferSize = CompilerServices.GetMaxShape([expr.Size])[0];
                var lifetime = new BufferLifetime(expr) { Memory = new(0, bufferSize) };
                _lifetimes.Add(expr, (lifetime, 0));
            }

            return default;
        }

        protected override Unit VisitLeafCall(Call expr)
        {
            foreach (var arg in expr.Arguments)
            {
                AcquireBuffer(arg);
            }

            return default;
        }

        private void AcquireBuffer(BaseExpr expr)
        {
            if (expr is IR.Tuple tuple)
            {
                foreach (var field in tuple.Fields)
                {
                    AcquireBuffer(field);
                }
            }
            else if (TryGetPhysicalBuffer(expr, out var buffer, out var physicalBuffer))
            {
                _buffers.Add(buffer);
                if (physicalBuffer.Start is None or Call { Target: IR.Buffers.AddressOf })
                {
                    ref var record = ref CollectionsMarshal.GetValueRefOrNullRef(_lifetimes, physicalBuffer);
                    record.RefCount++;
                }
            }
        }
    }

    private sealed class LifetimeRecoder : ExprWalker
    {
        private readonly Dictionary<TIR.PhysicalBuffer, (BufferLifetime Lifetime, int RefCount)> _lifetimes;
        private int _currentAge;

        public LifetimeRecoder(Dictionary<TIR.PhysicalBuffer, (BufferLifetime Lifetime, int RefCount)> lifetimes)
        {
            _lifetimes = lifetimes;
        }

        protected override Unit VisitLeafPhysicalBuffer(TIR.PhysicalBuffer expr)
        {
            if (expr.Start is None or Call { Target: IR.Buffers.AddressOf })
            {
                ref var record = ref CollectionsMarshal.GetValueRefOrNullRef(_lifetimes, expr);
                record.Lifetime.Time.Start = _currentAge;
            }

            return default;
        }

        protected override Unit VisitLeafCall(Call expr)
        {
            _currentAge++;
            foreach (var arg in expr.Arguments)
            {
                ReleaseBuffer(arg);
            }

            return default;
        }

        private void ReleaseBuffer(BaseExpr expr)
        {
            if (expr is IR.Tuple tuple)
            {
                foreach (var field in tuple.Fields)
                {
                    ReleaseBuffer(field);
                }
            }
            else if (TryGetPhysicalBuffer(expr, out _, out var physicalBuffer))
            {
                if (physicalBuffer.Start is None or Call { Target: IR.Buffers.AddressOf })
                {
                    ref var record = ref CollectionsMarshal.GetValueRefOrNullRef(_lifetimes, physicalBuffer);
                    if (--record.RefCount == 0)
                    {
                        record.Lifetime.Time.Stop = _currentAge;
                    }
                }
            }
        }
    }
}
