// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive;
using NetFabric.Hyperlinq;
using Nncase.IR;

namespace Nncase.Passes.Tile;

public static class ExprExtensions
{
    public static Expr Compose(this Expr expr, AffineMap map)
    {
        return expr.ReplaceDimsAndSymbols(map.Results, map.Symbols);
    }

    public static Expr ReplaceDimsAndSymbols(this Expr expr, Expr[] newDims, Expr[] newSymbols)
    {
        int i;
        switch (expr)
        {
            case TensorConst:
                return expr;
            case Var dimExpr when dimExpr.Name.StartsWith("d"):
                i = int.Parse(dimExpr.Name.Substring(1));
                if (i >= newDims.Length)
                {
                    return expr;
                }

                return newDims[i];
            case Var symExpr when symExpr.Name.StartsWith("s"):
                i = int.Parse(symExpr.Name.Substring(1));
                if (i >= newSymbols.Length)
                {
                    return expr;
                }

                return newSymbols[i];
            case Call { Target: IR.Math.Binary op } call:
                var lhs = ReplaceDimsAndSymbols(call[op.Parameters.First()], newDims, newSymbols);
                var rhs = ReplaceDimsAndSymbols(call[op.Parameters.Last()], newDims, newSymbols);
                return IR.F.Math.Binary(op.BinaryOp, lhs, rhs);
            case Call { Target: IR.Math.Unary { UnaryOp: UnaryOp.Neg } op } call:
                return IR.F.Math.Unary(op.UnaryOp, ReplaceDimsAndSymbols(call[op.Parameters.First()], newDims, newSymbols));
            case TIR.Range range:
                return new TIR.Range(ReplaceDimsAndSymbols(range.Start, newDims, newSymbols), ReplaceDimsAndSymbols(range.Stop, newDims, newSymbols), ReplaceDimsAndSymbols(range.Step, newDims, newSymbols));
            default:
                throw new InvalidOperationException("Unreachable");
        }
    }

    public static Expr[] Dims(int rank)
    {
        return Enumerable.Range(0, rank).Select(i => (Expr)new Var($"d{i}", DataTypes.Int32)).ToArray();
    }

    public static Expr[] Symbols(int rank)
    {
        return Enumerable.Range(0, rank).Select(i => (Expr)new Var($"s{i}", DataTypes.Int32)).ToArray();
    }

    public static string Display(this Expr expr)
    {
        switch (expr)
        {
            case Var var:
                return var.Name;
            case TensorConst @const:
                return @const.Value.ToScalar<int>().ToString();
            case Call { Target: IR.Math.Unary op } call:
                return op.UnaryOp switch
                {
                    UnaryOp.Neg => $"-{Display(call[op.Parameters.First()])}",
                    _ => throw new InvalidOperationException("Unreachable Unary Op"),
                };
            case Call { Target: IR.Math.Binary op } call:
                return op.BinaryOp switch
                {
                    BinaryOp.Add => $"{Display(call[op.Parameters.First()])} + {Display(call[op.Parameters.Last()])}",
                    BinaryOp.Mul => $"{Display(call[op.Parameters.First()])} * {Display(call[op.Parameters.Last()])}",
                    BinaryOp.Sub => $"{Display(call[op.Parameters.First()])} - {Display(call[op.Parameters.Last()])}",
                    BinaryOp.Div => $"{Display(call[op.Parameters.First()])} / {Display(call[op.Parameters.Last()])}",
                    BinaryOp.Mod => $"{Display(call[op.Parameters.First()])} % {Display(call[op.Parameters.Last()])}",
                    BinaryOp.FloorDiv => $"{Display(call[op.Parameters.First()])} // {Display(call[op.Parameters.Last()])}",
                    BinaryOp.CeilDiv => $"{Display(call[op.Parameters.First()])} \\\\ {Display(call[op.Parameters.Last()])}",
                    _ => throw new InvalidOperationException("Unreachable Binary Op"),
                };
            case TIR.Range rg:
                return $"({rg.Start.Display()}, {rg.Stop.Display()}, {rg.Step.Display()})";
            default:
                throw new InvalidOperationException("Unreachable Affine Expr");
        }
    }
}

public sealed class MapCloner : ExprCloner<Unit>
{
    private readonly IReadOnlyDictionary<Expr, Expr> _multiExprMap;

    public MapCloner(IReadOnlyDictionary<Expr, Expr> multiExprMap)
    {
        _multiExprMap = multiExprMap;
    }

    protected override Expr VisitLeafVar(Var expr, Unit context)
    {
        if (_multiExprMap.TryGetValue(expr, out var newVar))
        {
            return newVar;
        }

        throw new InvalidOperationException("Could not find var in map.");
    }
}

public class AffineMap
{
    public AffineMap(Expr[] dims, Expr[] symbols, Expr[] results)
    {
        Dims = dims;
        Symbols = symbols;
        Results = results;
    }

    public Expr[] Dims { get; set; }

    public Expr[] Symbols { get; set; }

    public Expr[] Results { get; }

    public static AffineMap ConstantMap(int value)
    {
        return new AffineMap(Array.Empty<Expr>(), Array.Empty<Expr>(), new[] { (Expr)value });
    }

    public static AffineMap PointMap(params int[] values)
    {
        return new AffineMap(Array.Empty<Expr>(), Array.Empty<Expr>(), values.Select(v => (Expr)v).ToArray());
    }

    public static AffineMap Identity(int rank)
    {
        var dims = Enumerable.Range(0, rank).Select(i => (Expr)new Var($"d{i}", DataTypes.Int32)).ToArray();
        return new AffineMap(dims, Array.Empty<Expr>(), dims);
    }

    public static AffineMap TransposeMap()
    {
        var dims = new[] { (Expr)new Var("d0", DataTypes.Int32), (Expr)new Var("d1", DataTypes.Int32) };
        return new AffineMap(dims, Array.Empty<Expr>(), new[] { dims[1], dims[0] });
    }

    public static AffineMap Empty()
    {
        return new AffineMap(Array.Empty<Expr>(), Array.Empty<Expr>(), Array.Empty<Expr>());
    }

    public static AffineMap FromCallable<T>(T func, int dimsNum, int symbsNum)
        where T : Delegate
    {
        var dims = Enumerable.Range(0, dimsNum).Select(i => (Expr)new Var($"d{i}", DataTypes.Int32)).ToArray();
        var symbols = Enumerable.Range(0, symbsNum).Select(i => (Expr)new Var($"s{i}", DataTypes.Int32)).ToArray();
        var funcParams = func.Method.GetParameters();
        object? results = null;
        if (funcParams.Length == 1 && funcParams[0].ParameterType.IsArray)
        {
            results = func.DynamicInvoke(new object[] { dims.Concat(symbols).ToArray() });
        }
        else
        {
            results = func.DynamicInvoke(dims.Concat(symbols).ToArray());
        }

        if (results is Expr[] ret)
        {
            return new AffineMap(dims, symbols, ret);
        }

        throw new NotSupportedException("Only Expr[] is supported.");
    }

    public AffineMap ReplaceDimsAndSymbols(Expr[] newDims, Expr[] newSymbols, int skipSymbols = 0)
    {
        var newResults = Results.Select(expr => expr.ReplaceDimsAndSymbols(newDims, newSymbols.Skip(skipSymbols).ToArray())).ToArray();
        return new AffineMap(newDims, newSymbols, newResults);
    }

    /// <summary>
    /// Y->Z compose X->Y => X->Z.
    /// </summary>
    public AffineMap Compose(AffineMap other)
    {
        if (Dims.Length != other.Results.Length)
        {
            throw new InvalidOperationException("Cannot compose AffineMaps with mismatching dimensions and results.");
        }

        var numDims = other.Dims.Length;
        var numSymbols = Symbols.Length + other.Symbols.Length;
        var newDims = ExprExtensions.Dims(numDims);
        var newSymbols = ExprExtensions.Symbols(numSymbols);

        var newMap = other.ReplaceDimsAndSymbols(newDims, newSymbols, Symbols.Length);
        var results = Results.Select(expr => expr.Compose(newMap)).ToArray();
        return new AffineMap(newMap.Dims, newMap.Symbols, results);
    }

    public AffineMap InversePermutation()
    {
        if (Symbols.Length != 0)
        {
            throw new InvalidOperationException("Cannot invert AffineMap with symbols.");
        }

        var foundDims = new int[Dims.Length];
        Array.Fill(foundDims, -1);

        for (int i = 0; i < Results.Length; i++)
        {
            if (Results[i] is { } dimExpr && foundDims[((TensorConst)dimExpr).Value.ToScalar<int>()] == -1)
            {
                foundDims[((TensorConst)dimExpr).Value.ToScalar<int>()] = i;
            }
        }

        if (foundDims.Any(d => d == -1))
        {
            return null!;
        }

        var results = foundDims.Select(i => Results[i]).ToArray();
        return new AffineMap(Results, Array.Empty<Expr>(), results);
    }

    public List<int> Eval(int[] dims, int[] symbols)
    {
        if (dims.Length != Dims.Length || symbols.Length != Symbols.Length)
        {
            throw new ArgumentException("Dimension and symbol arrays must match the map's dimensions and symbols.");
        }

        var feedDict = new Dictionary<Var, IValue>();
        foreach (var (first, second) in Dims.Zip(dims))
        {
            feedDict.Add((Var)first, Value.FromTensor(Tensor.FromScalar(second)));
        }

        foreach (var (first, second) in Symbols.Zip(symbols))
        {
            feedDict.Add((Var)first, Value.FromTensor(Tensor.FromScalar(second)));
        }

        return Results.Select(expr => expr.Evaluate(feedDict).AsTensor().ToScalar<int>()).ToList();
    }

    public Expr[] Apply(Expr[] parameters)
    {
        if (parameters.Length != Dims.Length + Symbols.Length)
        {
            throw new ArgumentException("Parameters must match the map's dimensions and symbols.");
        }

        Dictionary<Expr, Expr> map = new(ReferenceEqualityComparer.Instance);
        for (int i = 0; i < parameters.Length; i++)
        {
            map.Add(i < Dims.Length ? Dims[i] : Symbols[i - Dims.Length], parameters[i]);
        }

        var cloner = new MapCloner(map);

        return Results.Select(r => cloner.Clone(r, default)).ToArray();
    }

    public override string ToString()
    {
        var dims = string.Join(", ", Enumerable.Range(0, Dims.Length).Select(i => $"d{i}"));
        var syms = string.Join(", ", Enumerable.Range(0, Symbols.Length).Select(i => $"s{i}"));
        var results = string.Join(", ", Results.Select(expr => expr.Display()));

        return Symbols.Length == 0 ? $"({dims}) -> ({results})" : $"({dims})[{syms}] -> ({results})";
    }
}
