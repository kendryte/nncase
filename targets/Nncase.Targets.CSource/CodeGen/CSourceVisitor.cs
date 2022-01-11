using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using Nncase.IR;
using Nncase.TIR;

namespace Nncase.CodeGen;

/// <summary>
/// convert the type/op to c name
/// </summary>
internal static class NameConverter
{
    public static string toC(this DataType dataType) => dataType.ElemType switch
    {
        ElemType.Bool => "bool",
        ElemType.Int8 => "int8_t",
        ElemType.Int16 => "int16_t",
        ElemType.Int32 => "int32_t",
        ElemType.Int64 => "int64_t",
        ElemType.UInt8 => "uint8_t",
        ElemType.UInt16 => "uint16_t",
        ElemType.UInt32 => "uint32_t",
        ElemType.UInt64 => "uint64_t",
        // ElemType.Float16 => "float16_t",
        ElemType.Float32 => "float",
        ElemType.Float64 => "double",
        // ElemType.BFloat16 => "bfloat16_t",
        _ => throw new NotSupportedException($"{dataType}")
    };
}

/// <summary>
/// the c symbol define
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
/// collect the csymbol's parameter
/// </summary>
internal class CSymbolParamList : IParameterList<CSymbol>, IEnumerable<CSymbol>
{
    CSymbol[] Symbols;
    public CSymbolParamList(CSymbol[] symbols)
    {
        Symbols = symbols;
    }

    public CSymbol this[ParameterInfo parameter] => Symbols[parameter.Index];
    public CSymbol this[int index] => Symbols[index];

    public IEnumerator<CSymbol> GetEnumerator()
    {
        return ((IEnumerable<CSymbol>)Symbols).GetEnumerator();
    }

    IEnumerator IEnumerable.GetEnumerator()
    {
        return Symbols.GetEnumerator();
    }
}


/// <summary>
/// visitor for the build c source code, the expr vistor return (type string , name string)
/// </summary>
internal class CSourceHostBuildVisior : ExprFunctor<CSymbol, string>
{

    /// <summary>
    /// source writer .
    /// TODO we need the decl writer
    /// </summary>
    readonly IRPrinter.ScopeWriter Scope;

    /// <summary>
    /// symbols name memo
    /// </summary>
    readonly Dictionary<Expr, CSymbol> Symbols = new(new RecordRefComparer<Expr>());

    /// <summary>
    /// <see cref="CSourceHostBuildVisior"/>
    /// </summary>
    /// <param name="textWriter"></param>
    public CSourceHostBuildVisior(TextWriter textWriter)
    {
        Scope = new IRPrinter.ScopeWriter(textWriter);
        // insert some declare
        Scope.IndWriteLine("#include <stdint.h>");
    }

    /// <inheritdoc/>
    public override CSymbol Visit(Call expr)
    {
        if (Symbols.TryGetValue(expr, out var symbol)) { return symbol; }
        var target = Visit(expr.Target);
        var args = new CSymbolParamList(expr.Parameters.Select(Visit).ToArray());
        var type = VisitType(expr.CheckedType!);
        Scope.Push();
        switch (expr.Target)
        {
            case IR.Math.Binary:
                Scope.Append($"({args[0].Doc} {target.Doc} {args[1].Doc})");
                break;
            case Store:
                Scope.Append($"{args[Store.Handle].Doc}[{args[Store.Index].Doc}] = {args[Store.Value].Doc}");
                break;
            case Load:
                Scope.Append($"{args[Store.Handle].Doc}[{args[Store.Index].Doc}]");
                break;
            case IR.Tensors.Cast:
                Scope.Append($"(({type}){args[IR.Tensors.Cast.Input].Doc})");
                break;
            default:
                Scope.Append($"{target.Doc}({string.Join(", ", args.Select(x => x.Doc))})");
                break;
        }
        symbol = new(type, Scope.Pop());
        Symbols.Add(expr, symbol);
        return symbol;
    }

    /// <inheritdoc/>
    public override CSymbol Visit(Const expr)
    {
        if (Symbols.TryGetValue(expr, out var symbol)) { return symbol; }
        if (expr.CheckedType is TensorType ttype && ttype.IsScalar)
        {
            symbol = new(VisitType(ttype), new($"{expr}"));
        }
        else
        {
            throw new NotSupportedException($"Not Support {expr.CheckedType} Const");
        }
        Symbols.Add(expr, symbol);
        return symbol;
    }

    /// <inheritdoc/>
    public override CSymbol Visit(Function expr)
    {
        if (Symbols.TryGetValue(expr, out var symbol)) { return symbol; }
        var retType = VisitType(((CallableType)expr.CheckedType!).ReturnType);
        Scope.Push();
        // 1. Function signature
        Scope.IndWrite($"{retType} {expr.Name}({string.Join(", ", expr.Parameters.Select(Visit))}) {{");
        // 2. Function body
        using (Scope.IndentUp())
        {
            Scope.Append(Visit(expr.Body).Doc);
        }
        // 3. Function closing
        Scope.IndWrite("}");
        symbol = new(CallableTypeToPtr((CallableType)expr.CheckedType!, expr.Name), Scope.Pop());
        // 4. write whole code
        Scope.IndWrite(symbol.Doc);
        return symbol;
    }

    /// <inheritdoc/>
    public override CSymbol Visit(Op expr)
    {
        if (Symbols.TryGetValue(expr, out var symbol)) { return symbol; }
        symbol = new("Invalid Op", new(expr switch
        {
            IR.Math.Binary op => op.ToLiteral(),
            TIR.Store op => "Store",
            TIR.Load op => "Load",
            IR.Tensors.Cast op => op.NewType.toC(),
            _ => throw new NotSupportedException($"{expr.GetType().Name}")
        }));
        Symbols.Add(expr, symbol);
        return symbol;
    }

    /// <inheritdoc/>
    public override CSymbol Visit(Var expr)
    {
        if (Symbols.TryGetValue(expr, out var symbol)) { return symbol; }
        symbol = new(VisitType(expr.CheckedType!), new(expr.Name));
        Symbols.Add(expr, symbol);
        return symbol;
    }

    /// <summary>
    /// assgin the loop var better name.
    /// </summary>
    public CSymbol VisitLoopVar(Expr expr, string prefix = "")
    {
        if (Symbols.TryGetValue(expr, out var symbol)) { return symbol; }
        symbol = new(VisitType(expr.CheckedType!), new(Scope.GetUniqueLoopVarName(expr, prefix)));
        Symbols.Add(expr, symbol);
        return symbol;
    }

    /// <inheritdoc/>
    public override CSymbol Visit(For expr)
    {
        if (Symbols.TryGetValue(expr, out var symbol)) { return symbol; }
        Scope.Push();
        // 1. For Loop signature
        var loopVar = VisitLoopVar(expr.LoopVar);
        Scope.Append($"for ({loopVar} = {Visit(expr.Dom.Min).Doc}; {loopVar.Doc} < {Visit(expr.Dom.Max).Doc}; {loopVar.Doc}++) {{");
        // 2. For Body
        Scope.Append(Visit(expr.Body).Doc);
        // 3. For closing
        Scope.IndWrite("}");
        symbol = new(VisitType(expr.CheckedType!), Scope.Pop());
        Symbols.Add(expr, symbol);
        return symbol;
    }

    /// <inheritdoc/>
    public override CSymbol Visit(Sequential expr)
    {
        if (Symbols.TryGetValue(expr, out var symbol)) { return symbol; }
        Scope.Push();
        Scope.AppendLine("");
        using (Scope.IndentUp())
        {
            foreach (var i in Enumerable.Range(0, expr.Fields.Count))
            {
                if (i == expr.Fields.Count - 1 &&
                    expr.Fields[i].CheckedType is TensorType)
                {
                    Scope.IndWrite("return ");
                }
                else
                {
                    Scope.IndWrite(string.Empty);
                }
                Scope.Append(Visit(expr.Fields[i]).Doc);
                if (expr.Fields[i] is Call)
                {
                    Scope.AppendLine(";");
                }
                else
                {
                    Scope.AppendLine(string.Empty);
                }
            }
        }
        symbol = new(VisitType(expr.CheckedType!), Scope.Pop());
        Symbols.Add(expr, symbol);
        return symbol;
    }

    /// <inheritdoc/>
    public override CSymbol Visit(IfThenElse expr)
    {
        if (Symbols.TryGetValue(expr, out var symbol)) { return symbol; }
        Scope.Push();
        Scope.Append($"if({Visit(expr.Condition).Doc}) {{");
        Scope.Append(Visit(expr.Then).Doc);
        Scope.IndWrite("} else {");
        Scope.Append(Visit(expr.Else).Doc);
        Scope.IndWrite("}");
        symbol = new(VisitType(expr.CheckedType!), Scope.Pop());
        Symbols.Add(expr, symbol);
        return symbol;
    }

    /// <inheritdoc/>
    /// <example>
    /// void (*fun_ptr)(int)
    /// </example>
    public string CallableTypeToPtr(CallableType type, string name) => $"{VisitType(type.ReturnType)} (*{name}_ptr)({string.Join(",", type.Parameters.Select(VisitType))})";


    /// <inheritdoc/>
    public override string VisitType(TensorType type)
    {
        if (!type.IsScalar && type.DType.Lanes != 1)
        {
            throw new NotSupportedException($"{type}");
        }
        return type.DType.toC();
    }

    /// <inheritdoc/>
    public override string VisitType(HandleType type)
    {
        if (type.DType.Lanes != 1)
        {
            throw new NotSupportedException($"{type}");
        }
        return $"{type.DType.toC()}*";
    }

    /// <inheritdoc/>
    public override string VisitType(TupleType type) => type == TupleType.Void ?
      "void" :
      throw new InvalidProgramException($"The C Source Must Not Have TupleType {type}!");
}