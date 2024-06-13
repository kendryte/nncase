// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Google.OrTools.ConstraintSolver;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Affine;
using Isl = IntegerSetLibrary;

namespace Nncase.Schedule;

public static class TilingUtilities
{
    public static Isl.basic_map GetIdentityMap(int dims, string domainPrefix = "", string rangePrefix = "")
    {
        var space = Isl.space.unit(Isl.ctx.Instance);
        var s = space;
        if (domainPrefix.Any())
        {
            space = space.add_named_tuple(new Isl.id(Isl.ctx.Instance, domainPrefix), (uint)dims);
        }
        else
        {
            space = space.add_unnamed_tuple((uint)dims);
        }

        if (rangePrefix.Any())
        {
            space = space.add_named_tuple(new Isl.id(Isl.ctx.Instance, rangePrefix), (uint)dims);
        }
        else
        {
            space = space.add_unnamed_tuple((uint)dims);
        }

        return Isl.basic_map.identity(space);
    }

    public static int[] GetBufferShape(Expr buffer)
    {
        return buffer.CheckedType switch
        {
            TensorType t => t.Shape.ToValueArray(),
            DistributedType dt => Utilities.DistributedUtility.GetDividedTensorType(dt).Shape.ToValueArray(),
            _ => throw new NotSupportedException(),
        };
    }

    public static int[] InferDomainBounds(int[][] bufferShapes, AffineMap[] accessMaps)
    {
        var solver = new Solver("affineSolver");
        var converter = new AffineExprToIntExprConverter(solver);
        for (int i = 0; i < bufferShapes.Length; i++)
        {
            var shape = bufferShapes[i];
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
}
