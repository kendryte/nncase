﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using IntegerSetLibrary;
using NetFabric.Hyperlinq;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.F;
using Nncase.IR.Tensors;
using static Nncase.IR.F.Tensors;
using Isl = IntegerSetLibrary;

namespace Nncase.Utilities;

public static class ISLUtility
{
    public static Isl.set ToDomain(Shape shape, out HashSet<DimVar> dimVars)
    {
        dimVars = new(IR.ExprCollector.Collect(shape).OfType<DimVar>().ToArray());
        var constraints = dimVars.Select(d => $"{checked((int)d.Metadata.Range!.Value.Min)} <= {d.Name} <= {checked((int)d.Metadata.Range!.Value.Max)}").ToList();
        var dims = shape.Select((d, i) => $"d{i}").ToArray();
        for (int i = 0; i < dims.Length; i++)
        {
            constraints.Add($"{dims[i]} < {shape[i]}"); // note we can't assume the dims[i] >= 0, sometimes the shape will be 0, so that the dims[i] = -1.
        }

        return new Isl.set(Isl.ctx.Current, $"[{string.Join(',', dimVars)}] -> {{ [{string.Join(',', dims)}] : {string.Join(" and ", constraints)} }}");
    }

    public static Isl.set ToParametricDomain(Shape shape, out Dictionary<DimVar, Dimension> paramMap)
    {
        paramMap = new Dictionary<DimVar, Dimension>();
        var dims = new List<string>();
        var parameters = new List<string>();
        var constraints = new List<string>();
        for (int i = 0; i < shape.Rank; i++)
        {
            dims.Add($"d{i}");
            var dim = shape[i];
            if (dim is DimConst dimConst)
            {
                constraints.Add($"0 <= d{i} < {dimConst.Value}");
            }
            else
            {
                var range = dim.Metadata!.Range!;
                parameters.Add($"D{i}");
                var dimVar = new DimVar($"D{i}")
                {
                    Metadata = new()
                    {
                        Range = range,
                    },
                };
                paramMap.Add(dimVar, shape[i]);
                constraints.Add($"{range.Value.Min} <= D{i} <= {range.Value.Max}");
                constraints.Add($"0 <= d{i} < D{i}");
            }
        }

        return new Isl.set(Isl.ctx.Current, $"[{string.Join(',', parameters)}] -> {{ [{string.Join(',', dims)}] : {string.Join(" and ", constraints)} }}");
    }

    public static Dimension ToDimension(this Isl.pw_aff pa, IReadOnlyDictionary<string, Dimension> feedDict, string[]? dimNames = null)
    {
        var build = Isl.ast_build.from_context(dimNames is not null ? new Isl.set(Isl.ctx.Current, $"{{ [{string.Join(',', dimNames)}] }}") : pa.domain_space().universe_set());
        var astExpr = build.expr_from(pa);
        return ToDimension(astExpr, feedDict);
    }

    public static Dimension ToDimension(this Isl.ast_expr astExpr, IReadOnlyDictionary<string, Dimension> feedDict)
    {
        return new AstExprToExprConverter(feedDict).Visit(astExpr);
    }

    public static Isl.set ToSet(this Dimension dim, out HashSet<DimVar> dimVars)
    {
        dimVars = new(IR.ExprCollector.Collect(dim).OfType<DimVar>().ToArray());
        var constraints = dimVars.Select(d =>
        {
            if (d.Metadata.Range is ValueRange<double> range)
            {
                if (range.IsFull)
                {
                    return "true";
                }
                else
                {
                    return $"{checked((int)range.Min)} <= {d.Name} <= {checked((int)range.Max)}";
                }
            }

            return "true";
        }).ToArray();
        return new Isl.set(Isl.ctx.Current, $"[{string.Join(',', dimVars)}] -> {{ [{dim}] : {string.Join(" and ", constraints)} }}");
    }

    public static Isl.set ToSet(this ReadOnlySpan<Dimension> dims, out HashSet<DimVar> dimVars)
    {
        dimVars = new HashSet<DimVar>();
        var sets = dims.AsValueEnumerable().Select(d =>
        {
            var aff = ToSet(d, out var vars);
            return (aff, vars);
        }).ToArray();
        dimVars = new HashSet<DimVar>(sets.SelectMany(a => a.vars));
        return sets.Aggregate(new Isl.set(Isl.ctx.Current, "{ [] }"), (acc, value) => acc.flat_product(value.aff));
    }

    public static Dimension[] RoundTrip(this ReadOnlySpan<Dimension> dims)
    {
        var pma = dims.ToSet(out var dimVars).as_pw_multi_aff();
        var feedDict = dimVars.ToDictionary(v => v.Name, v => (Dimension)v);
        var build = Isl.ast_build.from_context(pma.domain_space().universe_set());

        var dimensions = new Dimension[dims.Length];
        for (int i = 0; i < dims.Length; i++)
        {
            var astExpr = build.expr_from(pma.at(i));
            dimensions[i] = astExpr.ToDimension(feedDict);
        }

        return dimensions;
    }
}

internal sealed class AstExprToExprConverter
{
    private readonly IReadOnlyDictionary<string, Dimension> _feedDict;

    public AstExprToExprConverter(IReadOnlyDictionary<string, Dimension> feedDict)
    {
        _feedDict = feedDict;
    }

    public Dimension Visit(Isl.ast_expr astExpr)
    {
        return astExpr.type() switch
        {
            Isl.ast_expr_type.id => VisitId(astExpr),
            Isl.ast_expr_type.int_ => astExpr.val().num_si(),
            Isl.ast_expr_type.op => astExpr.op_type() switch
            {
                Isl.ast_expr_op_type.add => Visit(astExpr.op_arg(0)) + Visit(astExpr.op_arg(1)),
                Isl.ast_expr_op_type.sub => Visit(astExpr.op_arg(0)) - Visit(astExpr.op_arg(1)),
                Isl.ast_expr_op_type.mul => Visit(astExpr.op_arg(0)) * Visit(astExpr.op_arg(1)),
                Isl.ast_expr_op_type.div => Visit(astExpr.op_arg(0)) / Visit(astExpr.op_arg(1)),
                Isl.ast_expr_op_type.select => VisitSelect(astExpr),
                Isl.ast_expr_op_type.min => Dimension.Min(Visit(astExpr.op_arg(0)), Visit(astExpr.op_arg(1))),
                Isl.ast_expr_op_type.minus => -Visit(astExpr.op_arg(0)),
                _ => throw new NotSupportedException($"Unsupported expr op type: {astExpr.op_type()}"),
            },
            _ => throw new NotSupportedException($"Unsupported expr type: {astExpr.type()}"),
        };
    }

    private Dimension VisitSelect(Isl.ast_expr select)
    {
        var cond = select.op_arg(0);
        return cond.type() switch
        {
            Isl.ast_expr_type.int_ => Visit(cond),
            Isl.ast_expr_type.op => cond.op_type() switch
            {
                Isl.ast_expr_op_type.eq => Dimension.Select(Visit(cond.op_arg(0)), Visit(cond.op_arg(1)), Visit(select.op_arg(1)), Visit(select.op_arg(2)), CompareOp.Equal),
                Isl.ast_expr_op_type.le => Dimension.Select(Visit(cond.op_arg(0)), Visit(cond.op_arg(1)), Visit(select.op_arg(1)), Visit(select.op_arg(2)), CompareOp.LowerOrEqual),
                Isl.ast_expr_op_type.lt => Dimension.Select(Visit(cond.op_arg(0)), Visit(cond.op_arg(1)), Visit(select.op_arg(1)), Visit(select.op_arg(2)), CompareOp.LowerThan),
                Isl.ast_expr_op_type.ge => Dimension.Select(Visit(cond.op_arg(0)), Visit(cond.op_arg(1)), Visit(select.op_arg(1)), Visit(select.op_arg(2)), CompareOp.GreaterOrEqual),
                Isl.ast_expr_op_type.gt => Dimension.Select(Visit(cond.op_arg(0)), Visit(cond.op_arg(1)), Visit(select.op_arg(1)), Visit(select.op_arg(2)), CompareOp.GreaterThan),
                _ => throw new NotSupportedException($"Unsupported select condition op type: {cond.op_type()}"),
            },
            _ => throw new NotSupportedException($"Unsupported select condition type: {cond.type()}"),
        };
    }

    private Dimension VisitId(ast_expr id)
    {
        return _feedDict[id.id().name()];
    }
}
