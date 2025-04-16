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

public sealed class LifetimeCollector
{
    public BufferLifetime[] Collect(Expr expr)
    {
        var lifetimes = new Dictionary<TIR.Buffer, (BufferLifetime Lifetime, int RefCount)>(ReferenceEqualityComparer.Instance);
        new BufferCollector(lifetimes).Visit(expr);
        new LifetimeRecoder(lifetimes).Visit(expr);
        ValidateZeroRefCounts(lifetimes);
        return lifetimes.Values.Select(x => x.Lifetime).ToArray();
    }

    private static bool TryGetBuffer(Expr expr, [MaybeNullWhen(false)] out TIR.Buffer buffer)
    {
        switch (expr)
        {
            case TIR.Buffer b:
                buffer = b;
                return true;
            default:
                break;
        }

        buffer = null;
        return false;
    }

    private static void ValidateZeroRefCounts(Dictionary<TIR.Buffer, (BufferLifetime Lifetime, int RefCount)> lifetimes)
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
        private readonly Dictionary<TIR.Buffer, (BufferLifetime Lifetime, int RefCount)> _lifetimes;

        public BufferCollector(Dictionary<TIR.Buffer, (BufferLifetime Lifetime, int RefCount)> lifetimes)
        {
            _lifetimes = lifetimes;
        }

        protected override Unit VisitLeafBuffer(TIR.Buffer expr)
        {
            if (expr.MemSpan.Start is None or Call { Target: IR.Buffers.DDrOf })
            {
                (var bufferSize, _) = TensorUtilities.GetTensorMaxSizeAndStrides(expr.CheckedTensorType, expr.DistributedType);
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

        private void AcquireBuffer(Expr expr)
        {
            if (expr is IR.Tuple tuple)
            {
                foreach (var field in tuple.Fields)
                {
                    AcquireBuffer(field);
                }
            }
            else if (TryGetBuffer(expr, out var buffer))
            {
                if (buffer.MemSpan.Start is None or Call { Target: IR.Buffers.DDrOf })
                {
                    ref var record = ref CollectionsMarshal.GetValueRefOrNullRef(_lifetimes, buffer);
                    record.RefCount++;
                }
            }
        }
    }

    private sealed class LifetimeRecoder : ExprWalker
    {
        private readonly Dictionary<TIR.Buffer, (BufferLifetime Lifetime, int RefCount)> _lifetimes;
        private int _currentAge;

        public LifetimeRecoder(Dictionary<TIR.Buffer, (BufferLifetime Lifetime, int RefCount)> lifetimes)
        {
            _lifetimes = lifetimes;
        }

        protected override Unit VisitLeafBuffer(TIR.Buffer expr)
        {
            if (expr.MemSpan.Start is None or Call { Target: IR.Buffers.DDrOf })
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

        private void ReleaseBuffer(Expr expr)
        {
            if (expr is IR.Tuple tuple)
            {
                foreach (var field in tuple.Fields)
                {
                    ReleaseBuffer(field);
                }
            }
            else if (TryGetBuffer(expr, out var buffer))
            {
                if (buffer.MemSpan.Start is None or Call { Target: IR.Buffers.DDrOf })
                {
                    ref var record = ref CollectionsMarshal.GetValueRefOrNullRef(_lifetimes, buffer);
                    if (--record.RefCount == 0)
                    {
                        record.Lifetime.Time.Stop = _currentAge;
                    }
                }
            }
        }
    }
}
