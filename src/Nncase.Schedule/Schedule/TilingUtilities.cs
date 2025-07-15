// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Google.OrTools.ConstraintSolver;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.Utilities;
using Isl = IntegerSetLibrary;

namespace Nncase.Schedule;

public static class TilingUtilities
{
    public static Shape GetBufferShape(Expr buffer, bool maxShape)
    {
        return buffer.CheckedType switch
        {
            TensorType t => maxShape ? CompilerServices.GetMaxShape(t.Shape) : t.Shape,
            DistributedType dt => Utilities.DistributedUtility.GetDividedTensorType(dt, maxShape).Shape,
            _ => throw new NotSupportedException(),
        };
    }

    public static (Isl.set DomainSet, bool[] DomainDynamic, long[] DomainBoundValues, Dimension[] DomainBoundExprs) InferDomainBounds(Expr[] bufferExprs, Isl.set[] shapeDomains, Isl.map[] accessMaps, HashSet<DimVar> dimVars)
    {
        var reversedAccessMaps = accessMaps.Zip(shapeDomains).Select(pair => pair.First.reverse().intersect_domain(pair.Second)).ToArray();
        Isl.map domainMap = null!;
        var shapeExprMap = new Dictionary<string, Dimension>();
        for (int i = 0; i < shapeDomains.Length; i++)
        {
            var reversedAccess = reversedAccessMaps[i];
            domainMap = domainMap is null ? reversedAccess : domainMap.flat_domain_product(reversedAccess!);
            for (int j = 0; j < shapeDomains[i].n_dim(); j++)
            {
                domainMap = domainMap.set_dim_name(Isl.dim_type.in_, (uint)(i + j), $"d{i}_{j}");
                shapeExprMap.Add($"d{i}_{j}", new IR.DimAt(new IR.Shapes.ShapeOf(bufferExprs[i]), j));
            }
        }

        var domainSet = domainMap.range();
        var domainBoundMpas = domainSet.max_multi_pw_aff();
        var domainDynamic = new bool[domainSet.n_dim()];
        var domainBoundValues = new long[domainSet.n_dim()];
        var domainBoundExprs = new Dimension[domainSet.n_dim()];

        for (int i = 0; i < domainSet.n_dim(); i++)
        {
            var boundMpa = domainBoundMpas.at(i);
            domainDynamic[i] = !boundMpa.is_cst();
            if (domainDynamic[i])
            {
                var dimExpr = ISLUtility.ToDimension(domainMap.max_multi_pw_aff().at(i), shapeExprMap, shapeExprMap.Keys.ToArray());
                dimExpr.Metadata = new()
                {
                    Range = new(boundMpa.min_val().num_si() + 1, boundMpa.max_val().num_si() + 1),
                };
                domainBoundExprs[i] = dimExpr;
                domainBoundValues[i] = boundMpa.max_val().num_si() + 1;
            }
            else
            {
                domainBoundExprs[i] = domainBoundValues[i] = boundMpa.max_val().num_si() + 1;
            }
        }

        return (domainSet, domainDynamic, domainBoundValues, domainBoundExprs);
    }

    public static long[] InferDomainBounds(long[][] bufferShapes, AffineMap[] accessMaps)
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
            throw new InvalidOperationException("Tiling bounds infer failed!");
        }

        var dims = dimVars.Select(x => solutionCollector.Value(0, x)).ToArray();
        return dims;
    }
}
