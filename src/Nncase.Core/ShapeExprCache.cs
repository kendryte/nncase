// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;

namespace Nncase;

public class ShapeExprCache
{
    public ShapeExprCache(IReadOnlyDictionary<Var, Expr[]> varMap, Dictionary<Expr, Expr>? cache = null)
    {
        VarMap = varMap;
        Cache = cache ?? new();
    }

    public static ShapeExprCache Default => new(new Dictionary<Var, Expr[]>(), new());

    public Dictionary<Expr, Expr> Cache { get; set; }

    public IReadOnlyDictionary<Var, Expr[]> VarMap { get; set; }

    public static implicit operator ShapeExprCache(Dictionary<Var, Expr[]> varMap) => new(varMap);

    public static ShapeExprCache operator +(ShapeExprCache cache, Dictionary<Var, Expr[]> varMap)
    {
        var newVarMap = cache.VarMap.Concat(varMap).ToDictionary(pair => pair.Key, pair => pair.Value);
        return new ShapeExprCache(newVarMap, cache.Cache);
    }

    public static ShapeExprCache operator +(Dictionary<Var, Expr[]> varMap, ShapeExprCache cache) => cache + varMap;

    public void Add(Expr expr, Expr shape)
    {
        Cache[expr] = shape;
    }
}
