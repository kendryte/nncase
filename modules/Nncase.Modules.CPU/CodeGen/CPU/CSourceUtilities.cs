// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.CommandLine;
using System.Globalization;
using DryIoc.ImTools;
using Nncase.Diagnostics;
using Nncase.IR.Math;

namespace Nncase.CodeGen.CPU;

internal static class CSourceUtilities
{
    public static string ContertBinary(Binary binary, CSymbol[] arguments)
    {
        var lhs = arguments[Binary.Lhs.Index].Name;
        var rhs = arguments[Binary.Rhs.Index].Name;
        string str;
        switch (binary.BinaryOp)
        {
            case BinaryOp.Add or BinaryOp.Sub or BinaryOp.Mul or BinaryOp.Div:
                str = $"({lhs} {binary.BinaryOp.ToC()} {rhs})";
                break;
            case BinaryOp.Min:
                str = $"std::min({lhs}, {rhs})";
                break;
            default:
                throw new NotSupportedException();
        }

        return str;
    }

    public static bool TryGetDivRem(string dim, out int div, out int rem)
    {
        div = 0;
        rem = 0;
        if (dim.IndexOf('?', System.StringComparison.CurrentCulture) is int s && dim.IndexOf(':', System.StringComparison.CurrentCulture) is int e && s != -1 && e != -1)
        {
            div = int.Parse(dim[(s + 1)..e].Trim());
            rem = int.Parse(dim[(e + 1)..^1].Trim());
            return true;
        }

        return false;
    }

    internal static string ContertUnary(Unary op, CSymbol[] arguments)
    {
        var input = arguments[Unary.Input.Index].Name;
        string str;
        switch (op.UnaryOp)
        {
            default:
                str = $"nncase_mt->{arguments[0].Type}_{nameof(Unary).ToLower(CultureInfo.CurrentCulture)}_{op.UnaryOp.ToString().ToLower(CultureInfo.CurrentCulture)}{input}";
                break;
        }

        return str;
    }

    internal static string ContertCompare(Compare op, CSymbol[] arguments)
    {
        var lhs = arguments[Compare.Lhs.Index].Name;
        var rhs = arguments[Compare.Rhs.Index].Name;
        string str = $"({lhs} {op.CompareOp.ToC()} {rhs})";
        return str;
    }

    internal static string ContertSelect(Select s, CSymbol[] arguments)
    {
        var p = arguments[Select.Predicate.Index].Name;
        var lhs = arguments[Select.TrueValue.Index].Name;
        var rhs = arguments[Select.FalseValue.Index].Name;
        string str = $"({p} ? {lhs} : {rhs})";
        return str;
    }
}
