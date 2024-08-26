// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive;
using Nncase;
using Nncase.IR;
using Nncase.Utilities;

namespace Nncase.Passes.BufferSchedule;

public class LifeTimeCollector : ExprVisitor<Unit, Unit>
{
    public event Action<LifeTimeCollector>? Alias;

    public int TimeStamp { get; private set; }

    public Dictionary<Expr, Interval> LifenessMap { get; } = new(ReferenceEqualityComparer.Instance);

    public virtual IReadOnlyDictionary<Expr, ScheduleBuffer> Collect(Expr expr)
    {
        Visit(expr);
        Update(expr); // avoid final call time interval size == 1.

        // TODO: open Alias
        Alias?.Invoke(this);
        var d = new Dictionary<Expr, ScheduleBuffer>(ReferenceEqualityComparer.Instance);
        int count = 0;
        foreach (var (k, v) in LifenessMap)
        {
            var name = k switch
            {
                Call c => c.Target.GetType().Name,
                Var va => va.Name,
                _ => k.GetType().Name,
            };
            var size = ComputeBufferSize(k, k.CheckedType, out var shape, out var stride);
            d.Add(k, new(name, count++, v, new(0, size), shape, stride, false));
        }

        return d;
    }

    protected override Unit DefaultVisitLeaf(Expr expr) => Unit.Default;

    protected override Unit VisitLeafCall(Call expr)
    {
        foreach (var arg in expr.Arguments)
        {
            Update(arg);
        }

        Update(expr);

        TimeStamp += 2;

        // note we will update tuple field on the next call.
        // foreach (var item in expr.Users.Where(e => e is not (BaseFunction or IR.Tuple)))
        // {
        //     Update(item);
        // }
        return Unit.Default;
    }

    protected virtual int ComputeBufferSize(Expr expr, IRType type, out int[] shape, out int[] stride)
    {
        shape = Array.Empty<int>();
        stride = Array.Empty<int>();
        var size = 0;
        if (type is TensorType tensorType)
        {
            shape = tensorType.Shape.ToValueArray();
            stride = TensorUtilities.GetStrides(shape);
            size = TensorUtilities.GetSize(shape, stride, tensorType.DType.SizeInBytes);
        }
        else if (type is DistributedType distributedType)
        {
            if (DistributedUtility.TryGetDividedTensorType(distributedType, out var tt))
            {
                shape = tt.Shape.ToValueArray();
                stride = TensorUtilities.GetStrides(shape);
                size = TensorUtilities.GetSize(shape, stride, tt.DType.SizeInBytes);
            }
            else
            {
                throw new NotSupportedException();
            }
        }
        else if (type is TupleType tupleType)
        {
            size = 0;
            foreach (var item in tupleType)
            {
                size += ComputeBufferSize(null!, item, out _, out _);
            }
        }

        return size;
    }

    protected virtual void Update(Expr expr)
    {
        if (expr is Const or None or Var)
        {
            return;
        }

        // boxing store
        if (expr is Call c && c.CheckedType is TensorType && c.Arguments[0].CheckedType is DistributedType)
        {
            return;
        }

        if (expr is IR.Tuple t)
        {
            foreach (var item in t.Fields)
            {
                Update(item);
            }

            return;
        }

        if (!LifenessMap.TryGetValue(expr, out var interval))
        {
            interval = new(TimeStamp, TimeStamp + 1);
        }
        else
        {
            interval.Stop = TimeStamp + 1;
        }

        LifenessMap[expr] = interval;
    }
}
