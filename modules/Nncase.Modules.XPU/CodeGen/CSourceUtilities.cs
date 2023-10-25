﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Globalization;
using Nncase.Diagnostics;
using Nncase.IR.Math;

namespace Nncase.CodeGen.XPU;

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
            default:
                throw new NotSupportedException();
        }

        return str;
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
}
