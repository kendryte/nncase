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

internal sealed class AffineTiler
{
    private readonly Grid _grid;
    private readonly int[] _dims;
    private readonly Expr[] _tempBuffers;
    private readonly ISequentialBuilder<TIR.For>[] _loopBuilders;
    private readonly Var[] _domainOffsets;
    private readonly Expr[] _domainExtents;

    public AffineTiler(Grid grid)
    {
        _grid = grid;
        _dims = InferDims();
        _tempBuffers = new Expr[grid.Buffers.Length];

        _loopBuilders = new ISequentialBuilder<TIR.For>[_dims.Length];
        _domainOffsets = new Var[_dims.Length];
        _domainExtents = new Expr[_dims.Length];
    }

    public Call Tile(IRModule module)
    {
        // 1. Solve schedule
        var schedule = SolveSchedule();

        // 2. Create loop builders
        for (int loop = 0; loop < _loopBuilders.Length; loop++)
        {
            var domain = schedule.Loops[loop].Domain.Offset.Position;
            var begin = 0;
            var end = begin + _dims[domain];
            var stride = schedule.Loops[loop].TileSize;
            _domainExtents[domain] = stride;
            _loopBuilders[loop] = T.ForLoop(out _domainOffsets[domain], (begin, end, stride), LoopMode.Serial, $"l{loop}");
        }

        var root = T.Sequential();
        ISequentialBuilder<Expr> cntBlock = root;

        // 2. Allocate temporal buffers
        // 2.1. Place 0
        cntBlock = AllocateTempBuffers(schedule.Places[0], cntBlock);

        // 2.2. Place 1..
        for (int loop = 0; loop < _loopBuilders.Length; loop++)
        {
            var place = loop + 1;
            var loopBuilder = _loopBuilders[loop];
            cntBlock.Body(loopBuilder);
            cntBlock = AllocateTempBuffers(schedule.Places[place], loopBuilder);
        }

        // 3. Nest compute body
        var bodyBuffers = new Expr[_grid.Buffers.Length];
        var bodyVarReplaces = new Dictionary<Expr, Expr>();
        for (int i = 0; i < bodyBuffers.Length; i++)
        {
            (bodyBuffers[i], cntBlock) = AllocateSubBuffer(cntBlock, _tempBuffers[i], schedule.BodyBufferViews[i]);
            bodyVarReplaces.Add(_grid.BodyParameters[i], bodyBuffers[i]);
        }

        var cloner = new ReplacingExprCloner(bodyVarReplaces);
        var nestBody = cloner.Clone(_grid.Body, default);
        cntBlock.Body(nestBody);

        // 4. Create PrimFunction
        var body = root.Build();
        var primFunc = new PrimFunction(_grid.ModuleKind, body, _grid.Buffers);
        var wrapper = new PrimFunctionWrapper(primFunc, _grid.Buffers.Length - 1);
        module.Add(primFunc);
        module.Add(wrapper);
        return new Call(wrapper, _grid.Buffers);
    }

    private ISequentialBuilder<Expr> AllocateTempBuffers(GridSchedule.Place place, ISequentialBuilder<Expr> sequential)
    {
        for (int i = 0; i < place.TemporalBuffers.Length; i++)
        {
            var tempBuffer = place.TemporalBuffers[i];
            (var bufferExpr, sequential) = AllocateSubBuffer(sequential, _grid.Buffers[tempBuffer.Buffer], tempBuffer.Subview);
            _tempBuffers[tempBuffer.Buffer] = bufferExpr;
        }

        return sequential;
    }

    private (Expr Buffer, ISequentialBuilder<Expr> NewSeq) AllocateSubBuffer(ISequentialBuilder<Expr> parentSeq, Expr parentBuffer, AffineMap accessMap)
    {
        var regions = accessMap.Results.AsValueEnumerable().Select(x => x.Apply(_domainOffsets, _domainExtents, null));
        var offset = new IR.Tuple(regions.Select(x => x.Offset).ToArray());
        var shape = new IR.Tuple(regions.Select(x => x.Extent).ToArray());
        var bufferExpr = IR.F.Buffer.AllocateBufferView(parentBuffer, offset, shape);
        var letExpr = T.Let(out var letVar, bufferExpr);
        parentSeq.Body(letExpr);
        return (letVar, letExpr);
    }

    private GridSchedule SolveSchedule()
    {
        var bufferShapes = _grid.Buffers.AsValueEnumerable().Select(x => x.CheckedShape.ToValueArray()).ToArray();
        var solver = new TilingSolver(_dims, bufferShapes, _grid.AccessMaps.ToArray());
        return solver.Solve();
    }

    private int[] InferDims()
    {
        var solver = new Solver("affineSolver");
        var converter = new AffineExprToIntExprConverter(solver);
        for (int i = 0; i < _grid.Buffers.Length; i++)
        {
            var shape = _grid.Buffers[i].CheckedShape.ToValueArray();
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
                v = _solver.MakeIntVar(1, int.MaxValue);
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
