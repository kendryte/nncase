// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

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

    public List<ScheduleBuffer> Collect(Function entry)
    {
        Visit(entry.Body);
        Alias();

        var l = new List<ScheduleBuffer>();
        foreach (var (k, v) in LifenessMap)
        {
            var name = k switch
            {
                Call c => c.Target.GetType().Name,
                Var va => va.Name,
                _ => k.GetType().Name,
            };

            var shape = k.CheckedShape.ToValueArray();
            var stride = TensorUtilities.GetStrides(shape);
            var size = TensorUtilities.GetSize(shape, stride, k.CheckedDataType.SizeInBytes);

            l.Add(new(name, v, new(0, size), shape, stride));
        }

        return l;
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
        if (expr is Const)
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
            interval.End += 1;
        }

        // advance the getitem buffer.
        if (expr is Call { Target: IR.Tensors.GetItem, Arguments: var args } call && args[0] is Call { CheckedType: TupleType })
        {
            interval.Start = LifenessMap[args[0]].Start;
        }

        LifenessMap[expr] = interval;
    }

    private void Alias()
    {
        // skip the call which output type is tuple.
        var calls = LifenessMap.Select(kv => kv.Key is Call { CheckedType: TupleType }).ToArray();
        foreach (var c in calls)
        {
            LifenessMap.Remove(c);
        }
    }
}
