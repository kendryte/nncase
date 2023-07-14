// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using NetFabric.Hyperlinq;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.Runtime;
using Nncase.TIR;

namespace Nncase.CodeGen;

/// <summary>
/// the c symbol define.
/// </summary>
internal struct CSymbol
{
    public string Type;
    public StringBuilder Doc;

    public CSymbol(string type, StringBuilder doc)
    {
        Type = type;
        Doc = doc;
    }

    public override string ToString() => $"{Type} {Doc}";
}

/// <summary>
/// convert the type/op to c name.
/// </summary>
internal static class NameConverter
{
    private static readonly Dictionary<PrimType, string> _primTypeToC = new()
    {
        { DataTypes.Boolean, "bool" },
        { DataTypes.Int8, "int8_t" },
        { DataTypes.Int16, "int16_t" },
        { DataTypes.Int32, "int32_t" },
        { DataTypes.Int64, "int64_t" },
        { DataTypes.UInt8, "uint8_t" },
        { DataTypes.UInt16, "uint16_t" },
        { DataTypes.UInt32, "uint32_t" },
        { DataTypes.UInt64, "uint64_t" },
        { DataTypes.Float32, "float" },
        { DataTypes.Float64, "double" },
    };

    public static string ToC(this PrimType primType) =>
        _primTypeToC[primType];

    public static string ToC(this DataType dataType) => dataType switch
    {
        PrimType ptype => ptype.ToC(),
        PointerType { ElemType: PrimType etype } => etype.ToC() + "*",
        _ => throw new NotSupportedException(dataType.ToString()),
    };
}

/// <summary>
/// collect the csymbol's parameter.
/// </summary>
internal class CSymbolParamList : IParameterList<CSymbol>, IEnumerable<CSymbol>
{
    private CSymbol[] _symbols;

    public CSymbolParamList(CSymbol[] symbols)
    {
        this._symbols = symbols;
    }

    public CSymbol this[ParameterInfo parameter] => _symbols[parameter.Index];

    public CSymbol this[int index] => _symbols[index];

    public IEnumerator<CSymbol> GetEnumerator()
    {
        return ((IEnumerable<CSymbol>)_symbols).GetEnumerator();
    }

    IEnumerator IEnumerable.GetEnumerator()
    {
        return _symbols.GetEnumerator();
    }
}

/// <summary>
/// visitor for the build c source code, the expr vistor return (type string , name string).
/// </summary>
internal class CSourceHostBuildVisior : ExprFunctor<CSymbol, string>
{
    /// <summary>
    /// source writer .
    /// TODO we need the decl writer.
    /// </summary>
    private readonly ScopeWriter _scope;

    /// <summary>
    /// symbols name memo.
    /// </summary>
    private readonly Dictionary<Expr, CSymbol> _symbols = new(ReferenceEqualityComparer.Instance);

    /// <summary>
    /// Initializes a new instance of the <see cref="CSourceHostBuildVisior"/> class.
    /// <see cref="CSourceHostBuildVisior"/>.
    /// </summary>
    /// <param name="textWriter">TextWriter.</param>
    public CSourceHostBuildVisior(TextWriter textWriter)
    {
        _scope = new ScopeWriter(textWriter);

        // insert some declare
        _scope.IndWriteLine(@"
#ifdef _WIN32
#define EXPORT_API __declspec(dllexport)
#else
#define EXPORT_API
#endif");
        _scope.IndWriteLine("#include <stdint.h>");
    }

    /// <example>
    /// void (*fun_ptr)(int).
    /// </example>
    public string CallableTypeToPtr(CallableType type, string name) => $"{VisitType(type.ReturnType)} (*{name}_ptr)({string.Join(",", type.Parameters.Select(VisitType))})";

    /// <inheritdoc/>
    public override string VisitType(TensorType type)
    {
        if (!type.IsScalar)
        {
            throw new NotSupportedException($"{type}");
        }

        return type.DType.ToC();
    }

    /// <inheritdoc/>
    public override string VisitType(TupleType type) => type == TupleType.Void ? "void" : throw new InvalidProgramException($"The C Source Must Not Have TupleType {type}!");

    /// <inheritdoc/>
    protected override CSymbol VisitCall(Call expr)
    {
        if (_symbols.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        var target = Visit(expr.Target);
        var args = new CSymbolParamList(expr.Arguments.AsValueEnumerable().Select(Visit).ToArray());
        var type = VisitType(expr.CheckedType!);
        _scope.Push();
        switch (expr.Target)
        {
            case IR.Math.Binary:
                _scope.Append($"({args[0].Doc} {target.Doc} {args[1].Doc})");
                break;
            case Store:
                _scope.Append($"{args[Store.Handle].Doc}[{args[Store.Index].Doc}] = {args[Store.Value].Doc}");
                break;
            case Load:
                _scope.Append($"{args[Store.Handle].Doc}[{args[Store.Index].Doc}]");
                break;
            case IR.Tensors.Cast:
                _scope.Append($"(({type}){args[IR.Tensors.Cast.Input].Doc})");
                break;
            default:
                _scope.Append($"{target.Doc}({string.Join(", ", args.Select(x => x.Doc))})");
                break;
        }

        symbol = new(type, _scope.Pop());
        _symbols.Add(expr, symbol);
        return symbol;
    }

    /// <inheritdoc/>
    protected override CSymbol VisitConst(Const expr)
    {
        if (_symbols.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        if (expr.CheckedType is TensorType ttype && ttype.IsScalar)
        {
            var literal = $"{expr}" switch
            {
                "True" => "1",
                "False" => "0",
                var x => x,
            };
            symbol = new(VisitType(ttype), new(literal));
        }
        else
        {
            throw new NotSupportedException($"Not Support {expr.CheckedType} Const");
        }

        _symbols.Add(expr, symbol);
        return symbol;
    }

    /// <inheritdoc/>
    protected override CSymbol VisitPrimFunction(PrimFunction expr)
    {
        if (_symbols.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        var retType = VisitType(((CallableType)expr.CheckedType!).ReturnType);
        _scope.Push();

        // 1. Function signature
        _scope.IndWrite($"EXPORT_API {retType} {expr.Name}({string.Join(", ", expr.Parameters.AsValueEnumerable().Select(Visit).ToArray())}) {{");

        // 2. Function body
        using (_scope.IndentUp())
        {
            _scope.Append(Visit(expr.Body).Doc);
        }

        // 3. Function closing
        _scope.IndWrite("}");
        symbol = new(CallableTypeToPtr((CallableType)expr.CheckedType!, expr.Name), _scope.Pop());

        // 4. write whole code
        _scope.IndWrite(symbol.Doc);
        return symbol;
    }

    /// <inheritdoc/>
    protected override CSymbol VisitOp(Op expr)
    {
        if (_symbols.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        symbol = new("Invalid Op", new(expr switch
        {
            IR.Math.Binary op => op.BinaryOp switch
            {
                BinaryOp.Add => "+",
                BinaryOp.Sub => "-",
                BinaryOp.Mul => "*",
                BinaryOp.Div => "/",
                BinaryOp.Mod => "%",
                _ => throw new ArgumentOutOfRangeException(op.BinaryOp.ToString()),
            },
            TIR.Store op => "Store",
            TIR.Load op => "Load",
            IR.Tensors.Cast op => op.NewType.ToC(),
            _ => throw new NotSupportedException($"{expr.GetType().Name}"),
        }));
        _symbols.Add(expr, symbol);
        return symbol;
    }

    /// <inheritdoc/>
    protected override CSymbol VisitVar(Var expr)
    {
        if (_symbols.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        var isymbol = _scope.GetUniqueVarSymbol(expr);
        symbol = new(VisitType(expr.CheckedType!), isymbol.Span);
        _symbols.Add(expr, symbol);
        return symbol;
    }

    /// <inheritdoc/>
    protected override CSymbol VisitFor(For expr)
    {
        if (_symbols.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        _scope.Push();

        // 1. For Loop signature
        var loopVar = Visit(expr.LoopVar);
        _scope.Append($"for ({loopVar} = {Visit(expr.Domain.Start).Doc}; {loopVar.Doc} < {Visit(expr.Domain.Stop).Doc}; {loopVar.Doc}+={expr.Domain.Step}) {{");

        // 2. For Body
        _scope.Append(Visit(expr.Body).Doc);

        // 3. For closing
        _scope.IndWrite("}");
        symbol = new(VisitType(expr.CheckedType!), _scope.Pop());
        _symbols.Add(expr, symbol);
        return symbol;
    }

    /// <inheritdoc/>
    protected override CSymbol VisitSequential(Sequential expr)
    {
        if (_symbols.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        _scope.Push();
        _scope.AppendLine(string.Empty);
        using (_scope.IndentUp())
        {
            foreach (var i in Enumerable.Range(0, expr.Count))
            {
                if (i == expr.Count - 1 &&
                    expr.Fields[i].CheckedType is TensorType)
                {
                    _scope.IndWrite("return ");
                }
                else
                {
                    _scope.IndWrite(string.Empty);
                }

                _scope.Append(Visit(expr.Fields[i]).Doc);
                if (expr.Fields[i] is Call)
                {
                    _scope.AppendLine(";");
                }
                else
                {
                    _scope.AppendLine(string.Empty);
                }
            }
        }

        symbol = new(VisitType(expr.CheckedType!), _scope.Pop());
        _symbols.Add(expr, symbol);
        return symbol;
    }

    /// <inheritdoc/>
    protected override CSymbol VisitIfThenElse(IfThenElse expr)
    {
        if (_symbols.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        _scope.Push();
        _scope.Append($"if({Visit(expr.Condition).Doc}) {{");
        _scope.Append(Visit(expr.Then).Doc);
        _scope.IndWrite("} else {");
        _scope.Append(Visit(expr.Else).Doc);
        _scope.IndWrite("}");
        symbol = new(VisitType(expr.CheckedType!), _scope.Pop());
        _symbols.Add(expr, symbol);
        return symbol;
    }
}
