// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Diagnostics;
using System.Linq;
using System.Numerics;
using System.Reactive;
using System.Text;
using System.Threading.Tasks;
using Google.OrTools.ConstraintSolver;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.TIR;
using Nncase.TIR.Builders;
using Nncase.Utilities;

namespace Nncase.Schedule;

public static class Tiling
{
    public static Call AutoTile(Grid grid)
    {
        var dims = InferDims(grid.Buffers, grid.AccessMaps);
        var schedule = SolveSchedule(grid, dims);

        var tempBuffers = new Expr[grid.Buffers.Length];
        var root = T.Sequential();
        var lastSeq = root;
        AllocateTempBuffers(tempBuffers, grid.Buffers, schedule.Places[0], lastSeq);

        var loopBuilders = new TIR.Builders.ISequentialBuilder<TIR.For>[dims.Length];
        for (int loop = 0; loop < loopBuilders.Length; loop++)
        {
            var loopBuilder = T.ForLoop(out var loopVar, (1, 1), LoopMode.Serial);
        }

        return new Call(null);
    }

    public static Call AutoTile()
    {
        var input = Const.FromTensor(Tensor.FromScalar(1, new[] { 3, 16, 16 }));
        var rank = input.CheckedShape.Rank;

        var grid = IR.F.Affine.Grid()
            .Read(input, AffineMap.Identity(rank), out var inTile)
            .Write(TIR.T.CreateBuffer(input.CheckedTensorType, TIR.MemoryLocation.Data, out _), AffineMap.Identity(rank), out _)
            .Body(IR.F.Math.Unary(UnaryOp.Abs, inTile))
            .Build();
        return AutoTile(grid);
    }

    private static void AllocateTempBuffers(Expr[] tempBuffers, ReadOnlySpan<Expr> buffers, GridSchedule.Place place, ISequentialBuilder<Sequential> sequential)
    {
        foreach (var tempBuffer in place.TemporalBuffers)
        {
            // var region = tempBuffer.Subview.Results[0].Apply()
            // var bufferExpr = IR.F.Buffer.AllocateBufferView(buffers[tempBuffer.Buffer], );
        }
    }

    private static GridSchedule SolveSchedule(Grid grid, int[] dims)
    {
        var bufferShapes = grid.Buffers.AsValueEnumerable().Select(x => x.CheckedShape.ToValueArray()).ToArray();
        var solver = new TilingSolver(dims, bufferShapes, grid.AccessMaps.ToArray());
        return solver.Solve();
    }

    private static int[] InferDims(ReadOnlySpan<Expr> buffers, ReadOnlySpan<AffineMap> accessMaps)
    {
        var solver = new Solver("affineSolver");
        var converter = new AffineExprToIntExprConverter(solver);
        for (int i = 0; i < buffers.Length; i++)
        {
            var shape = buffers[i].CheckedShape.ToValueArray();
            var results = accessMaps[i].Results;
            for (int j = 0; j < results.Length; j++)
            {
                var extent = results[j].Extent;
                var expr = converter.Visit(extent);
                solver.Add(expr == shape[j]);
            }
        }

        var dimVars = accessMaps[0].Domains.AsValueEnumerable().Select(x => (IntVar)converter.Visit(x.Extent)).ToArray();
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
