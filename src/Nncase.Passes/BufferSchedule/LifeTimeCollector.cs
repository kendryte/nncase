// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive;
using Nncase;
using Nncase.IR;

namespace Nncase.Passes.BufferSchedule;

internal sealed class LifeTimeCollector : ExprVisitor<Unit, Unit>
{
    public int TimeStamp { get; private set; }

    public Dictionary<Expr, TimeInterval> LifenessMap { get; } = new(ReferenceEqualityComparer.Instance);

    public IReadOnlyDictionary<Expr, ScheduleBuffer> Collect(Function entry)
    {
        Visit(entry.Body);

        var d = new Dictionary<Expr, ScheduleBuffer>(ReferenceEqualityComparer.Instance);
        foreach (var (k, v) in LifenessMap)
        {
            var name = k switch
            {
                Call c => c.Target.GetType().Name,
                Var va => va.Name,
                _ => k.GetType().Name,
            };
            var size = GetSize(k.CheckedType, out var shape, out var stride);

            d.Add(k, new(name, v, new(0, size), shape, stride, IsInPlace(k)));
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

        TimeStamp += 1;

        return Unit.Default;
    }

    private void Update(Expr expr)
    {
        if (expr is (Const or None))
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
            interval.Death = TimeStamp + 1;
        }

        LifenessMap[expr] = interval;
    }

    private bool IsInPlace(Expr expr)
    {
        if (expr is Call { Target: IR.Tensors.Reshape } callReshape)
        {
            return true;
        }

        if (expr is Call { Target: IR.Tensors.Concat } concatCall && concatCall.Arguments[0] is IR.Tuple concatTuple)
        {
            return true;
        }

        if (expr is Call { Target: IR.Tensors.Split } splitCall)
        {
            return true;
        }

        return false;
    }

    private int GetSize(IRType type, out int[] shape, out int[] stride)
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
        else if (type is TupleType tupleType)
        {
            size = 0;
            foreach (var item in tupleType)
            {
                size += GetSize(item, out _, out _);
            }
        }

        return size;
    }

}
