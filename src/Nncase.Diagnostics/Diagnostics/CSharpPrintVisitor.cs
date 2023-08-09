// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.TIR;
using Nncase.Utilities;

namespace Nncase.Diagnostics;

internal sealed class CSharpPrintVisitor : ExprFunctor<string, string>
{
    private readonly ScopeWriter _scope;
    private readonly Dictionary<Expr, string> _names = new Dictionary<Expr, string>(ReferenceEqualityComparer.Instance);
    private readonly BinaryWriter _constWriter;
    private readonly bool _randConst;
    private int _localId;

    public CSharpPrintVisitor(TextWriter textWriter, BinaryWriter constWriter, int indent_level, bool randConst, bool withHeader = true)
    {
        _scope = new(textWriter, indent_level);
        _constWriter = constWriter;
        _randConst = randConst;
        if (withHeader)
        {
            _scope.IndWriteLine("Tensor GetD<T>(System.IO.BinaryReader __reader, long __start, int __size, params int[] __shape)");
            _scope.IndWriteLine("where T : unmanaged, IEquatable<T>");
            _scope.IndWriteLine("{");
            using (_scope.IndentUp())
            {
                if (_randConst)
                {
                    _scope.IndWriteLine("var buffer = new byte[__size];");
                    _scope.IndWriteLine("Testing.RandGenerator.NextBytes(buffer);");
                    _scope.IndWriteLine("return Tensor.FromBytes<T>(buffer, __shape);");
                }
                else
                {
                    _scope.IndWriteLine("__reader.BaseStream.Seek(__start, System.IO.SeekOrigin.Begin);");
                    _scope.IndWriteLine("return Tensor.FromBytes<T>(__reader.ReadBytes(__size), __shape);");
                }
            }

            _scope.IndWriteLine("}");
            if (_randConst)
            {
                _scope.IndWriteLine("using var vD = new System.IO.BinaryReader(new System.IO.MemoryStream());");
            }
            else
            {
                _scope.IndWriteLine("using var vD = new System.IO.BinaryReader(System.IO.File.OpenRead(??));");
            }
        }
    }

    /// <inheritdoc/>
    public override string VisitType(AnyType type) => "any";

    /// <inheritdoc/>
    public override string VisitType(CallableType type) =>
        $"({string.Join(", ", type.Parameters.Select(VisitType))}) -> {VisitType(type.ReturnType)}";

    /// <inheritdoc/>
    public override string VisitType(InvalidType type) => $"invalid:{type.Reason}";

    /// <inheritdoc/>
    public override string VisitType(NoneType type) => $"";

    /// <inheritdoc/>
    public override string VisitType(TensorType type) => type.DType switch
    {
        PrimType ptype => ptype.GetDisplayName() + (type.Shape.IsScalar ? string.Empty : type.Shape.ToString()),
        PointerType { ElemType: PrimType etype } => $"*{etype.GetDisplayName()}",
        ValueType => $"{type.DType.ToString()}",
        _ => throw new NotSupportedException(type.DType.GetType().Name),
    };

    /// <inheritdoc/>
    public override string VisitType(TupleType type) =>
        $"({string.Join(", ", type.Fields.Select(VisitType))})";

    /// <inheritdoc/>
    protected override string VisitCall(Call expr)
    {
        if (_names.TryGetValue(expr, out var name))
        {
            return name;
        }

        var target = Visit(expr.Target);
        var args = expr.Arguments.AsValueEnumerable().Select(Visit).ToArray();
        name = AllocateTempVar(expr);
        _scope.IndWrite($"var {name} = new Call({target}, new Expr[] {{{string.Join(", ", args)}}})");
        AppendCheckedType(expr.CheckedType);
        return name;
    }

    /// <inheritdoc/>
    protected override string VisitConst(Const expr)
    {
        if (_names.TryGetValue(expr, out var name))
        {
            return name;
        }

        name = GetCSharpConst(expr);
        _names.Add(expr, name);
        return name;
    }

    /// <inheritdoc/>
    protected override string VisitFunction(Function expr)
    {
        if (_names.TryGetValue(expr, out var name))
        {
            return name;
        }

        name = AllocateTempVar(expr);
        _scope.Push();

        // 1. functionv var
        _scope.IndWrite($"Function {name}");
        AppendCheckedType(expr.CheckedType);

        // 2. Function body
        _scope.IndWriteLine("{");
        using (_scope.IndentUp())
        {
            var body = Visit(expr.Body);
            _scope.IndWriteLine($"{name} = new Function(\"{expr.Name}\", {body}, new Var[] {{{StringUtility.Join(", ", expr.Parameters.AsValueEnumerable().Select(Visit))}}});");
        }

        // 3. Function signature
        _scope.IndWriteLine("}");
        _scope.Append(_scope.Pop());
        return name;
    }

    protected override string VisitFusion(Fusion expr)
    {
        if (_names.TryGetValue(expr, out var name))
        {
            return name;
        }

        name = AllocateTempVar(expr);

        _scope.IndWrite($"Fusion {name}");
        AppendCheckedType(expr.CheckedType);
        _scope.Push();
        _scope.IndWriteLine("{");
        using (_scope.IndentUp())
        {
            var body_builder = new StringBuilder();
            string body;
            using (var body_writer = new StringWriter(body_builder))
            {
                var visitor = new CSharpPrintVisitor(body_writer, _constWriter, _scope.IndentLevel, _randConst, false) { _localId = _localId };
                body = visitor.Visit(expr.Body);
                _scope.Append(body_writer.ToString());
            }

            _scope.IndWriteLine($"{name} = new Fusion(\"{expr.Name}\", \"{expr.ModuleKind}\", {body}, new Var[] {{{StringUtility.Join(", ", expr.Parameters.AsValueEnumerable().Select(Visit))}}});");
        }

        _scope.IndWriteLine("}");
        _scope.Append(_scope.Pop());
        return name;
    }

    /// <inheritdoc/>
    protected override string VisitOp(Op expr)
    {
        if (_names.TryGetValue(expr, out var name))
        {
            return name;
        }

        name = $"new {expr.GetType().Name}({expr.DisplayProperty()})";
        _names.Add(expr, name);
        return name;
    }

    /// <inheritdoc/>
    protected override string VisitTuple(IR.Tuple expr)
    {
        if (_names.TryGetValue(expr, out var name))
        {
            return name;
        }

        var fields = expr.Fields.AsValueEnumerable().Select(Visit).ToArray();
        name = AllocateTempVar(expr);
        _scope.IndWrite($"var {name} = new IR.Tuple(new Expr[]{{{string.Join(", ", fields)}}})");
        AppendCheckedType(expr.CheckedType);
        _scope.IndWriteLine();
        return name;
    }

    /// <inheritdoc/>
    protected override string VisitVar(Var expr)
    {
        if (_names.TryGetValue(expr, out var name))
        {
            return name;
        }

        name = AllocateTempVar(expr);
        _scope.IndWriteLine($"var {name} = new Var(\"{expr.Name}\", {GetCSharpIRType(expr.TypeAnnotation)});");
        return name;
    }

    /// <inheritdoc/>
    protected override string VisitNone(None expr)
    {
        if (_names.TryGetValue(expr, out var name))
        {
            return name;
        }

        name = $"None.Default";
        _names.Add(expr, name);
        return name;
    }

    /// <inheritdoc/>
    protected override string VisitMarker(Marker expr)
    {
        if (_names.TryGetValue(expr, out var name))
        {
            return name;
        }

        var target = Visit(expr.Target);
        var attr = Visit(expr.Attribute);
        name = AllocateTempVar(expr);
        _scope.IndWrite($"var {name} = new Marker(\"{expr.Name}\",{target},{attr})");
        AppendCheckedType(expr.CheckedType);
        return name;
    }

    private string AllocateTempVar(Expr expr)
    {
        var name = $"v{_localId++}";
        _names.Add(expr, name);
        return name;
    }

    private void AppendCheckedType(IRType? type, string end = "", bool hasNewLine = true)
    {
        if (type is not null)
        {
            if (hasNewLine)
            {
                _scope.AppendLine($"; // {VisitType(type)}{end}");
            }
            else
            {
                _scope.Append($"; // {VisitType(type)}{end}");
            }
        }
        else
        {
            _scope.Append(";\n");
        }
    }

    private string GetCSharpIRType(IRType type) => type switch
    {
        TensorType ttype => $"new TensorType({ttype.DType.GetCSharpName()}, new [] {{{string.Join(",", ttype.Shape)}}})",
        TupleType ttype => $"new TupleType({string.Join(",", ttype.Fields)})",
        AnyType => "AnyType.Default",
        NoneType => "NoneType.Default",
        _ => "AnyType.Default",
    };

    private string GetArrayComma(Shape shape) => shape.Rank > 0 ? string.Join(string.Empty, Enumerable.Repeat<char>(',', shape.Rank - 1)) : string.Empty;

    private string GetCSharpConstFromFile(TensorConst tc)
    {
        var start = _constWriter.BaseStream.Position;
        _constWriter.Write(tc.Value.BytesBuffer);
        var end = _randConst ? tc.Value.BytesBuffer.Length : _constWriter.BaseStream.Position;
        var size = end - start;
        var shape = tc.Value.Shape.IsScalar ? string.Empty : $", {string.Join(",", tc.Value.Shape.ToValueArray())}";
        return $"GetD<{tc.Value.ElementType.GetBuiltInName()}>(vD, {start}, {size}{shape})";
    }

    private string GetCSharpConst(Const @const) => @const switch
    {
        TensorConst tc => tc.Value.ElementType switch
        {
            PrimType primType => tc.Value.Shape switch
            {
                Shape { IsScalar: true } => tc.Value.GetArrayString(false),
                Shape x when x.Size <= 8 => $"new {primType.GetBuiltInName()}[{GetArrayComma(x)}]{tc.Value.GetArrayString(false)}",
                _ => GetCSharpConstFromFile(tc),
            },
            ValueType valueType => GetCSharpConstFromFile(tc),
            _ => "NotSupport",
        },
        TupleConst tc => $"new TupleConst(new Const[] {{{string.Join(",", tc.Value.Select(x => GetCSharpConst(Const.FromValue(x))))}}})",
        _ => throw new ArgumentOutOfRangeException(@const.GetType().Name),
    };
}
