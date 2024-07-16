// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Reactive;
using System.Text;
using System.Threading.Tasks;
using Google.OrTools.ConstraintSolver;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.TIR;
using Nncase.TIR.Builders;

namespace Nncase.Schedule;

internal sealed record AffineTilerMemo(IRArray<Shape> BufferShapes, IRArray<int> DomainBounds, IRArray<AffineMap> AffineMaps, Type OpType, int ElemSize)
{
}

internal sealed class AffineTiler
{
    private readonly Grid _grid;
    private readonly int[] _domainBounds;

    public AffineTiler(Grid grid, ITargetOptions targetOptions)
    {
        _grid = grid;
        TargetOptions = targetOptions;
        _domainBounds = InferDomainBounds();
    }

    public ITargetOptions TargetOptions { get; }

    public Call Tile(IRModule module, Dictionary<AffineTilerMemo, GridSchedule> memo)
    {
        // 1. Solve schedule
        if (_grid.Body[0] is not Call { Target: Op op })
        {
            throw new InvalidOperationException("body is not call");
        }

        var schedule = SolveSchedule(op, memo);
        var loopBuilders = new ISequentialBuilder<TIR.For>[schedule.Loops.Length];
        var domainOffsets = new Var[schedule.Loops.Length];
        var domainExtents = new Expr[schedule.Loops.Length];

        // 2. Create nested loop builders
        for (int loop = 0; loop < loopBuilders.Length; loop++)
        {
            var begin = 0ul;
            var end = begin + (ulong)schedule.Loops[loop].Stop;
            var stride = (ulong)schedule.Loops[loop].Stride;
            loopBuilders[loop] = T.ForLoop(out domainOffsets[schedule.Loops[loop].Domain.Offset.Position], (begin, end, stride), LoopMode.Serial, schedule.Loops[loop].Name);
            domainExtents[loop] = stride;
        }

        var rootBuilder = T.Sequential();
        ISequentialBuilder<Expr> cntBuilder = rootBuilder;

        // 3. Allocate temporal buffers
        // 3.1. Place at root
        var bufferScope = new Dictionary<int, List<Expr>>();
        for (int i = 0; i < _grid.Buffers.Length; i++)
        {
            bufferScope[i] = new() { _grid.Buffers[i] };
        }

        cntBuilder = AllocateTempBuffers(schedule.Places[0], cntBuilder, bufferScope, domainOffsets, domainExtents);

        // 2.2. Place 1..
        for (int loop = 0; loop < loopBuilders.Length; loop++)
        {
            var place = loop + 1;
            var loopBuilder = loopBuilders[loop];
            cntBuilder.Body(loopBuilder);
            if (place < schedule.Places.Length)
            {
                cntBuilder = AllocateTempBuffers(schedule.Places[place], loopBuilder, bufferScope, domainOffsets, domainExtents);
            }
            else
            {
                cntBuilder = loopBuilder;
            }
        }

        // 3. inner computation.
        var bodyBuffers = new Expr[_grid.Buffers.Length];
        var bufferOfVars = new Expr[_grid.Reads.Length + 1];
        var typehints = new IRType[_grid.Reads.Length + 1];
        var bodyVarReplaces = new Dictionary<Expr, Expr>();
        var bufferOfReplaces = new Dictionary<Expr, Expr>();
        for (int i = 0; i < bodyBuffers.Length; i++)
        {
            (bodyBuffers[i], cntBuilder) = AllocateSubBuffer(cntBuilder, bufferScope[i].Last(), schedule.BodyBufferViews[i], domainOffsets, domainExtents);
            bodyVarReplaces.Add(_grid.BodyParameters[i], bodyBuffers[i]);
        }

        for (int i = 0; i < _grid.Reads.Length; i++)
        {
            bufferOfVars[i] = AllocateBufferOf(_grid.Buffers[i], i);
            typehints[i] = _grid.Buffers[i].CheckedType;
            bufferOfReplaces.Add(_grid.Buffers[i], bufferOfVars[i]);
        }

        bufferOfVars[^1] = AllocateOutputBuffer(_grid.Buffers[^1]);
        bufferOfReplaces.Add(_grid.Buffers[^1], bufferOfVars[^1]);
        typehints[^1] = _grid.Buffers[^1].CheckedType;
        {
            var results = schedule.DomainMap.Apply(domainOffsets.Select(o => IR.F.Tensors.Cast(o, DataTypes.Int64)).ToArray(), domainExtents);
            for (int i = 0; i < schedule.DomainMap.Results.Length; i++)
            {
                bodyVarReplaces.Add(new AffineDim(i), results[i].Start);
            }
        }

        var nestBody = new ReplacingExprCloner(bodyVarReplaces).Clone(_grid.Body, default);
        cntBuilder.Body(nestBody);

        // 4. Create PrimFunction
        var body = rootBuilder.Build();
        body = new ReplacingExprCloner(bufferOfReplaces).Clone(body, default);
        var primFunc = new PrimFunction(_grid.ModuleKind, body, bufferOfVars);
        var wrapper = new PrimFunctionWrapper(primFunc, _grid.Buffers.Length - 1, typehints);

        // module.Add(primFunc);
        // module.Add(wrapper);
        return new Call(wrapper, _grid.Reads);
    }

    private TIR.Buffer AllocateBufferOf(Expr expr, int i)
    {
        if (expr is IR.Buffers.BufferOf bufof)
        {
            var ttype = bufof.Input.CheckedType switch
            {
                TensorType t => t,
                DistributedType dt => Utilities.DistributedUtility.GetDividedTensorType(dt),
                _ => throw new NotSupportedException(),
            };

            return T.AttachBuffer(None.Default, ttype, MemoryLocation.Input, 1, out _, "buffer_" + i.ToString());
        }

        throw new NotSupportedException();
    }

    private TIR.Buffer AllocateOutputBuffer(Expr expr)
    {
        if (expr is Call { Target: IR.Buffers.Uninitialized } c)
        {
            var ttype = c.CheckedType switch
            {
                TensorType t => t,
                DistributedType dt => Utilities.DistributedUtility.GetDividedTensorType(dt),
                _ => throw new NotSupportedException(),
            };

            return T.AttachBuffer(None.Default, ttype, MemoryLocation.Output, 1, out _, "buffer_out");
        }

        throw new NotSupportedException();
    }

    private ISequentialBuilder<Expr> AllocateTempBuffers(GridSchedule.Place place, ISequentialBuilder<Expr> sequential, Dictionary<int, List<Expr>> bufferScope, Var[] domainOffsets, Expr[] domainExtents)
    {
        for (int i = 0; i < place.TemporalBuffers.Length; i++)
        {
            var tempBuffer = place.TemporalBuffers[i];
            (var subBuffer, sequential) = AllocateSubBuffer(sequential, bufferScope[tempBuffer.Buffer].Last(), tempBuffer.Subview, domainOffsets, domainExtents);
            bufferScope[tempBuffer.Buffer].Add(subBuffer);
        }

        return sequential;
    }

    private (Expr Buffer, ISequentialBuilder<Expr> NewSeq) AllocateSubBuffer(ISequentialBuilder<Expr> builder, Expr parentBuffer, AffineMap accessMap, Var[] domainOffsets, Expr[] domainExtents)
    {
        var regions = accessMap.Results.AsValueEnumerable().Select(x => x.Apply(domainOffsets.ToArray(), domainExtents)).ToArray();
        var offset = new IR.Tuple(regions.Select(x => x.Offset).ToArray());
        var shape = new IR.Tuple(regions.Select(x => x.Extent).ToArray());
        var bufferExpr = IR.F.Buffer.BufferSubview(parentBuffer, offset, shape);
        string name = parentBuffer switch
        {
            IR.Buffers.BufferOf { Input: Var v } => v.Name + "_sub",
            TIR.Buffer b => b.Name + "_sub",
            Var v => v.Name + "_sub",
            Call { Target: IR.Buffers.Uninitialized } => "out",
            _ => "let",
        };

        var letBuilder = T.Let(out var letVar, bufferExpr, name);
        builder.Body(letBuilder);
        return (letVar, letBuilder);
    }

    private GridSchedule SolveSchedule(Op op, Dictionary<AffineTilerMemo, GridSchedule> memo)
    {
        var bufferShapes = new Shape[_grid.Buffers.Length];
        for (int i = 0; i < _grid.Buffers.Length; i++)
        {
            var shape = _grid.Buffers[i].CheckedType switch
            {
                TensorType t => t.Shape.ToValueArray(),
                DistributedType dt => Utilities.DistributedUtility.GetDividedTensorType(dt).Shape.ToValueArray(),
                _ => throw new NotSupportedException(),
            };
            bufferShapes[i] = new Shape(shape);
        }

        int elemSize = _grid.Buffers[0].CheckedDataType.SizeInBytes;
        var originalDomain = _grid.AccessMaps[0].Domains.ToArray().Select(d => d.Offset).ToArray();
        var key = new AffineTilerMemo(bufferShapes, _domainBounds, _grid.AccessMaps.ToArray(), op.GetType(), elemSize);
        if (!memo.TryGetValue(key, out var schedule))
        {
            var solver = new TilingSolver(TargetOptions);
            schedule = solver.Solve(_domainBounds, bufferShapes.Select(x => x.ToValueArray()).ToArray(), originalDomain, _grid.AccessMaps.ToArray(), op, elemSize);
            memo.Add(key, schedule);
            return schedule;
        }
        else
        {
            System.Console.WriteLine("use cached schedule");
            return schedule;
        }
    }

    private int[] InferDomainBounds()
    {
        var solver = new Solver("affineSolver");
        var converter = new AffineExprToIntExprConverter(solver);
        for (int i = 0; i < _grid.Buffers.Length; i++)
        {
            var shape = _grid.Buffers[i].CheckedType switch
            {
                TensorType t => t.Shape.ToValueArray(),
                DistributedType dt => Utilities.DistributedUtility.GetDividedTensorType(dt).Shape.ToValueArray(),
                _ => throw new NotSupportedException(),
            };
            var results = _grid.AccessMaps[i].Results;
            for (int j = 0; j < results.Length; j++)
            {
                var extent = results[j].Extent;
                var expr = converter.Visit(extent);
                solver.Add(expr == shape[j]);
            }
        }

        var dimVars = _grid.AccessMaps[0].Domains.AsValueEnumerable().Select(x => (IntVar)converter.Visit(x.Extent)).ToArray();
        var db = solver.MakePhase(dimVars, Solver.CHOOSE_FIRST_UNBOUND, Solver.ASSIGN_MIN_VALUE);
        var solutionCollector = solver.MakeFirstSolutionCollector();
        solutionCollector.Add(dimVars);
        solver.Solve(db, solutionCollector);

        if (solutionCollector.SolutionCount() < 1)
        {
            throw new InvalidOperationException();
        }

        var dims = dimVars.Select(x => (int)solutionCollector.Value(0, x)).ToArray();
        return dims;
    }

    private sealed class AffineExprToIntExprConverter : ExprVisitor<IntExpr, Unit>
    {
        private readonly Solver _solver;
        private readonly Dictionary<int, IntVar> _extents = new();

        public AffineExprToIntExprConverter(Solver solver)
        {
            _solver = solver;
        }

        protected override IntExpr VisitLeafAffineExtent(AffineExtent expr)
        {
            if (!_extents.TryGetValue(expr.Position, out var v))
            {
                v = _solver.MakeIntVar(1, int.MaxValue, $"d{expr.Position}_v");
                _extents.Add(expr.Position, v);
            }

            return v;
        }

        protected override IntExpr VisitLeafAffineConstant(AffineConstant expr) =>
            _solver.MakeIntConst(expr.Value);

        protected override IntExpr VisitLeafAffineAddBinary(AffineAddBinary expr) =>
            ExprMemo[expr.Lhs] + ExprMemo[expr.Rhs];

        protected override IntExpr VisitLeafAffineMulBinary(AffineMulBinary expr) =>
            ExprMemo[expr.Lhs] * ExprMemo[expr.Rhs];

        protected override IntExpr VisitLeafAffineDivBinary(AffineDivBinary expr) =>
            expr.BinaryOp switch
            {
                AffineDivBinaryOp.FloorDiv => _solver.MakeDiv(ExprMemo[expr.Lhs], ExprMemo[expr.Rhs]),
                AffineDivBinaryOp.CeilDiv => ExprMemo[expr.Lhs].CeilDiv(ExprMemo[expr.Rhs]),
                AffineDivBinaryOp.Mod => _solver.MakeModulo(ExprMemo[expr.Lhs], ExprMemo[expr.Rhs]),
                _ => throw new UnreachableException(),
            };
    }
}
