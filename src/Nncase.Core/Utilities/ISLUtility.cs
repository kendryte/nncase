// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using IntegerSetLibrary;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.F;
using Nncase.IR.Tensors;
using static Nncase.IR.F.Tensors;
using Isl = IntegerSetLibrary;

namespace Nncase.Utilities;

public static class ISLUtility
{
    public static Isl.set AsDomain(Shape shape)
    {
        var param = new List<string>();
        var dims = new List<string>();
        var constraints = new List<string>();
        for (int i = 0; i < shape.Rank; i++)
        {
            var dim = shape[i];
            switch (dim)
            {
                case DimConst c:
                    constraints.Add($"0 <= d{i} < {c.Value}");
                    break;
                case DimVar v:
                    var paramName = v.Name;
                    param.Add(paramName);
                    constraints.Add($"0 <= d{i} < {paramName}");
                    var rg = dim.Metadata.Range!;
                    constraints.Add($"{(int)rg.Value.Min} <= {paramName} <= {(int)rg.Value.Max}");
                    break;
            }

            dims.Add($"d{i}");
        }

        return new Isl.set(Isl.ctx.Current, $"[{string.Join(", ", param)}] -> {{ [{string.Join(", ", dims)}] : {string.Join(" and ", constraints)} }}");
    }

    public static Dimension AsDimension(this Isl.pw_aff pa, IReadOnlyDictionary<string, Dimension> feedDict)
    {
        var build = Isl.ast_build.from_context(pa.domain_space().universe_set());
        var astExpr = build.expr_from(pa);
        return AsDimension(astExpr, feedDict);
    }

    public static Dimension AsDimension(this Isl.ast_expr astExpr, IReadOnlyDictionary<string, Dimension> feedDict)
    {
        return new AstExprToExprConverter(feedDict).Visit(astExpr);
    }

    public static Isl.pw_multi_aff AsPwMultiAff(this Dimension dim, out HashSet<DimVar> dimVars)
    {
        dimVars = new(IR.ExprCollector.Collect(dim).OfType<DimVar>().ToArray());
        return new Isl.pw_multi_aff(Isl.ctx.Current, $"[{string.Join(',', dimVars)}] -> {{ [{dim}] }}");
    }

    public static Isl.pw_multi_aff AsPwMultiAff(this Shape shape, out HashSet<DimVar> dimVars)
    {
        dimVars = new HashSet<DimVar>();
        var affs = shape.Select(d =>
        {
            var aff = AsPwMultiAff(d, out var vars);
            return (aff, vars);
        }).ToArray();
        dimVars = new HashSet<DimVar>(affs.SelectMany(a => a.vars));
        return affs.Aggregate(new Isl.pw_multi_aff(Isl.ctx.Current, "{ [] }"), (acc, value) => acc.flat_range_product(value.aff));
    }

    public static Shape Simplify(Shape shape)
    {
        var pma = shape.AsPwMultiAff(out var dimVars);
        pma = pma.set_tuple_id(Isl.dim_type.in_, "dummy");
        var feedDict = dimVars.ToDictionary(v => v.Name, v => (Dimension)v);
        var build = Isl.ast_build.from_context(new Isl.set(ctx.Current, "{ dummy[] }"));
        var access = build.access_from(pma);
        var dimensions = new Dimension[shape.Rank];
        for (int i = 0; i < shape.Rank; i++)
        {
            dimensions[i] = access.op_arg(1 + i).AsDimension(feedDict);
        }

        return new RankedShape(dimensions);
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
                Isl.ast_expr_op_type.select => Dimension.Select(Visit(astExpr.op_arg(0)), 1, Visit(astExpr.op_arg(1)), Visit(astExpr.op_arg(2))),
                _ => throw new NotSupportedException($"Unsupported expr op type: {astExpr.op_type()}"),
            },
            _ => throw new NotSupportedException($"Unsupported expr type: {astExpr.type()}"),
        };
    }

    private Dimension VisitId(ast_expr id)
    {
        return _feedDict[id.id().name()];
    }
}
