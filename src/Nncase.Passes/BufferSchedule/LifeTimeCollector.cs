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

public class LifeTimeUpdater : ExprFunctor<Unit, Unit, LifeTimeUpdater.Context>
{
    protected override Unit DefaultVisit(Expr expr, Context context) => default;

    protected override Unit VisitTuple(IR.Tuple expr, Context context)
    {
        foreach (var item in expr.Fields)
        {
            if (item is IR.Tuple tp)
            {
                Visit(tp, context);
            }
            else if (item is Call c)
            {
                PerformUpdate(c, context);
            }
        }

        return default;
    }

    protected override Unit VisitCall(Call expr, Context context)
    {
        foreach (var item in expr.Arguments)
        {
            if (item is IR.Tuple tp)
            {
                Visit(tp, context);
            }
            else if (item is Call c)
            {
                PerformUpdate(c, context);
            }
        }

        PerformUpdate(expr, context);
        return default;
    }

    protected void PerformUpdate(Expr expr, Context context)
    {
        var (livenessMap, timeStamp) = context;
        if (!livenessMap.TryGetValue(expr, out var interval))
        {
            interval = new(timeStamp, timeStamp + 1);
        }
        else
        {
            interval.Stop = timeStamp + 1;
        }

        livenessMap[expr] = interval;
    }

    public sealed record Context(Dictionary<Expr, Interval> LivenessMap, int TimeStamp)
    {
    }
}

public class BufferSizeCalculator : ExprFunctor<BufferSizeCalculator.Result, BufferSizeCalculator.Result>
{
    public override Result DefaultVisitType(IRType type) => throw new NotSupportedException();

    public override Result VisitType(TensorType type)
    {
        var shape = type.Shape.ToValueArray();
        var stride = TensorUtilities.GetStrides(shape);
        return new(TensorUtilities.GetSize(shape, stride, type.DType.SizeInBytes), shape, stride);
    }

    public override Result VisitType(DistributedType distributedType)
    {
        if (DistributedUtility.TryGetDividedTensorType(distributedType, out var tt))
        {
            var shape = tt.Shape.ToValueArray();
            var stride = TensorUtilities.GetStrides(shape);
            var size = TensorUtilities.GetSize(shape, stride, tt.DType.SizeInBytes);
            return new(size, shape, stride);
        }
        else
        {
            throw new NotSupportedException();
        }
    }

    public override Result VisitType(TupleType tupleType)
    {
        var size = 0;
        foreach (var item in tupleType)
        {
            size += VisitType(item).Size;
        }

        return new(size, Array.Empty<int>(), Array.Empty<int>());
    }

    protected override Result VisitCall(Call expr)
    {
        if (expr.Target is IR.Tensors.GetItem)
        {
            if (expr.Arguments[1] is TensorConst tc && tc.Value.Shape.IsScalar)
            {
                var res = VisitType(expr.CheckedType);
                return new(0, res.Shape, res.Stride);
            }
            else
            {
                throw new NotSupportedException("getItem index is not const scalar!");
            }
        }

        return VisitType(expr.CheckedType);
    }

    public sealed record Result(int Size, int[] Shape, int[] Stride)
    {
        public static readonly Result Empty = new(0, Array.Empty<int>(), Array.Empty<int>());
    }
}

public class LifeTimeCollector : ExprVisitor<Unit, Unit>
{
    public LifeTimeCollector(LifeTimeUpdater updater, BufferSizeCalculator calculator)
    {
        Updater = updater;
        Calculator = calculator;
        FinishedCollect = null!;
    }

    public event EventHandler FinishedCollect;

    public int TimeStamp { get; private set; }

    public Dictionary<Expr, Interval> LivenessMap { get; } = new(ReferenceEqualityComparer.Instance);

    public LifeTimeUpdater Updater { get; }

    public BufferSizeCalculator Calculator { get; }

    public IReadOnlyDictionary<Expr, ScheduleBuffer> Collect(Expr expr)
    {
        Visit(expr);
        Updater.Visit(expr, new(LivenessMap, TimeStamp)); // avoid final call time interval size == 1.

        FinishedCollect?.Invoke(this, EventArgs.Empty); // custom some liveness.
        var d = new Dictionary<Expr, ScheduleBuffer>(ReferenceEqualityComparer.Instance);
        int count = 0;
        foreach (var (k, v) in LivenessMap)
        {
            var name = k switch
            {
                Call c => c.Target switch
                {
                    Op op => $"{op.GetType().Name}({op.DisplayProperty()})",
                    Callable cb => $"{cb.Name}@{cb.ModuleKind}",
                    _ => $"{c.Target.GetType().Name}",
                },
                Var va => va.Name,
                _ => k.GetType().Name,
            };

            var rest = Calculator.Visit(k);
            d.Add(k, new(name, count++, v, new(0, rest.Size), rest.Shape, rest.Stride, false));
        }

        return d;
    }

    protected override Unit DefaultVisitLeaf(Expr expr) => default;

    protected override Unit VisitLeafCall(Call expr)
    {
        Updater.Visit(expr, new(LivenessMap, TimeStamp));
        TimeStamp += 2;
        return default;
    }
}
