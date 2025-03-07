// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive;
using Nncase;
using Nncase.IR;
using Nncase.Passes.Transforms;
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

/// <summary>
/// default BufferSizeCalculator. already support optimize memory move for get_item.
/// </summary>
public class BufferSizeCalculator : ExprFunctor<BufferSizeCalculator.Result, BufferSizeCalculator.Result>
{
    public override Result DefaultVisitType(IRType type) => throw new NotSupportedException();

    public override Result VisitType(TensorType type)
    {
        var maxShape = CompilerServices.GetMaxShape(type.Shape);
        var stride = TensorUtilities.GetStrides(maxShape);
        return new(TensorUtilities.GetSize(maxShape, stride, type.DType.SizeInBytes), type.Shape, stride);
    }

    public override Result VisitType(DistributedType distributedType)
    {
        if (DistributedUtility.TryGetDividedTensorType(distributedType, out var tt))
        {
            var maxShape = CompilerServices.GetMaxShape(tt.Shape);
            var stride = TensorUtilities.GetStrides(maxShape);
            var size = TensorUtilities.GetSize(maxShape, stride, tt.DType.SizeInBytes);
            return new(size, tt.Shape, stride);
        }
        else
        {
            throw new NotSupportedException();
        }
    }

    public override Result VisitType(TupleType tupleType)
    {
        long size = 0;
        foreach (var item in tupleType)
        {
            size += VisitType(item).MaxSize;
        }

        return new(size, Array.Empty<long>(), Array.Empty<long>());
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
        }

        return VisitType(expr.CheckedType);
    }

    public sealed record Result(long MaxSize, Shape Shape, long[] Stride)
    {
        public static readonly Result Empty = new(0, Shape.Invalid, Array.Empty<long>());
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

    public HashSet<Var> DimVars { get; } = new(ReferenceEqualityComparer.Instance);

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
            d.Add(k, new(name, count++, v, new(0, rest.MaxSize), rest.Shape, rest.Stride, false));
            UpdateDimVars(rest);
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

    private void UpdateDimVars(BufferSizeCalculator.Result result)
    {
        new DimVarUpdater(DimVars).Visit(result.Shape);
    }

    private sealed class DimVarUpdater : ExprWalker
    {
        private readonly HashSet<Var> _dimVars;

        public DimVarUpdater(HashSet<Var> dimVars)
        {
            _dimVars = dimVars;
        }

        protected override Unit VisitLeafVar(Var expr)
        {
            _dimVars.Add(expr);
            return default;
        }
    }
}
