﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive;
using System.Text;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.IR;

namespace Nncase.CodeGen.StackVM;

internal class TextSnippet
{
    private readonly List<TextSnippet> _inputSnippets = new List<TextSnippet>();

    public TextSnippet(Expr expr, Symbol beginSymbol, Symbol endSymbol)
    {
        Expr = expr;
        BeginSymbol = beginSymbol;
        EndSymbol = endSymbol;
        Writer = new BinaryWriter(Text, Encoding.UTF8, leaveOpen: true);
        Emitter = new StackVMEmitter(Writer);
    }

    public Expr Expr { get; }

    public MemoryStream Text { get; } = new MemoryStream();

    public BinaryWriter Writer { get; }

    public StackVMEmitter Emitter { get; }

    public List<SymbolRef> SymbolRefs { get; } = new List<SymbolRef>();

    public List<FunctionRef> FunctionRefs { get; } = new List<FunctionRef>();

    public IReadOnlyList<TextSnippet> InputSnippets => _inputSnippets;

    public Symbol BeginSymbol { get; }

    public Symbol EndSymbol { get; }

    public int UseCount { get; private set; }

    public int MaxUserParameters { get; set; } = 1;

    public bool OutputInLocal => UseCount > 1 || MaxUserParameters > 1;

    public int RefCount { get; set; }

    public void AddInput(TextSnippet input)
    {
        _inputSnippets.Add(input);
        input.UseCount++;
    }
}

internal class CodeGenContext
{
    private readonly List<TextSnippet> _textSnippets = new List<TextSnippet>();
    private readonly HashSet<ModuleType> _custom_call_modules = new();

    public CodeGenContext(BinaryWriter rdataWriter)
    {
        RdataWriter = rdataWriter;
    }

    public BinaryWriter RdataWriter { get; }

    public Dictionary<DataType, Symbol> DataTypes { get; } = new Dictionary<DataType, Symbol>();

    public IReadOnlyList<TextSnippet> TextSnippets => _textSnippets;

    public IReadOnlySet<ModuleType> CustomCallModules => _custom_call_modules;

    public void AddTextSnippet(TextSnippet textSnippet)
    {
        _textSnippets.Add(textSnippet);
    }

    public void AddCustomCallModule(ModuleType moduleType)
    {
        _custom_call_modules.Add(moduleType);
    }
}

internal partial class CodeGenVisitor : ExprVisitor<TextSnippet, IRType>
{
    private const int _alignment = 8;
    private const byte _rdataGpid = 0;

    private readonly BaseFunction _function;
    private readonly CodeGenContext _context;

    private TextSnippet? _currentTextSnippet;

    public CodeGenVisitor(BaseFunction function, CodeGenContext context)
    {
        _function = function;
        _context = context;
    }

    private TextSnippet CurrentTextSnippet => _currentTextSnippet!;

    private StackVMEmitter Emitter => CurrentTextSnippet.Emitter;

    protected override TextSnippet VisitLeafConst(Const expr)
    {
        if (expr is TensorConst tc)
        {
            return Visit(tc, tc.Value);
        }
        else
        {
            return Visit(new IR.Tuple(((TupleConst)expr).Value.Select(x => Const.FromValue(x)).ToArray()));
        }
    }

    protected override TextSnippet VisitLeafNone(None expr)
    {
        var snippet = BeginTextSnippet(expr);
        Emitter.LdNull();
        return snippet;
    }

    protected override TextSnippet VisitLeafVar(Var expr)
    {
        var snippet = BeginTextSnippet(expr);
        var varIndex = ((Function)_function).Parameters.IndexOf(expr);
        if (varIndex < 0)
        {
            throw new InvalidOperationException($"Can't find var {expr.Name} in CodeGen");
        }

        Emitter.Ldarg((ushort)varIndex);
        return snippet;
    }

    protected override TextSnippet VisitLeafTuple(IR.Tuple expr)
    {
        var snippet = BeginTextSnippet(expr);
        foreach (var field in expr.Fields.ToArray().Reverse())
        {
            var inputSnippet = Visit(field);
            inputSnippet.MaxUserParameters = Math.Max(inputSnippet.MaxUserParameters, expr.Fields.Length);
            snippet.AddInput(inputSnippet);
        }

        Emitter.LdcI4(expr.Count);
        Emitter.LdTuple();
        return snippet;
    }

    protected override TextSnippet VisitFunction(Function expr)
    {
        if (ReferenceEquals(expr, _function))
        {
            return Visit(expr.Body);
        }
        else
        {
            return null!;
        }
    }

    protected override TextSnippet VisitPrimFunctionWrapper(PrimFunctionWrapper expr)
    {
        if (ReferenceEquals(expr, _function))
        {
            var snippet = BeginTextSnippet(expr);
            for (int i = expr.ParametersCount - 1; i >= 0; i--)
            {
                Emitter.Ldarg((ushort)i);
            }

            LdFunctionId(expr.Target);
            Emitter.ExtCall(checked((ushort)expr.ParameterTypes.Count()), true);
            return snippet;
        }
        else
        {
            return null!;
        }
    }

    protected override TextSnippet VisitLeafOp(Op expr)
    {
        return null!;
    }

    protected override TextSnippet VisitLeafCall(Call expr)
    {
        var snippet = BeginTextSnippet(expr);
        foreach (var param in expr.Arguments.ToArray().Reverse())
        {
            var paramSnippet = Visit(param);
            paramSnippet.MaxUserParameters = Math.Max(paramSnippet.MaxUserParameters, expr.Arguments.Length);
            snippet.AddInput(paramSnippet);
        }

        if (expr.Target is CustomOp custom_op)
        {
            _context.AddCustomCallModule(custom_op.ModuleType);
            Emitter.CusCall(custom_op.RegisteredName, custom_op.SerializeFields(), checked((ushort)expr.Arguments.Length));
        }
        else if (expr.Target is Op op)
        {
            EmitTensorCall(op);
        }
        else if (expr.Target is PrimFunctionWrapper wrapper)
        {
            LdFunctionId(wrapper.Target);
            Emitter.ExtCall(checked((ushort)wrapper.ParameterTypes.Count()), true);
        }
        else if (expr.Target is Function func)
        {
            LdFunctionId(func);
            Emitter.ExtCall(checked((ushort)func.Parameters.Length), false);
        }
        else
        {
            throw new NotSupportedException(expr.Target.GetType().Name);
        }

        return snippet;
    }

    protected override TextSnippet VisitIf(If expr)
    {
        return VisitLeafIf(expr);
    }

    /// <summary>
    /// Composition of if:
    /// 1. (Condition)
    /// 2. BrFalse
    /// {
    ///     3. Then
    ///     4. Br(AfterElse)
    /// }
    /// {
    ///     5. Else
    /// }
    /// 6. EndSnippet.
    /// </summary>
    /// <param name="if">If expr.</param>
    /// <returns>TextSnippet.</returns>
    protected override TextSnippet VisitLeafIf(If @if)
    {
        var condSnippet = Visit(@if.Condition);
        condSnippet.Emitter.LdScalar();
        var brFalse = BeginTextSnippet(@if);
        brFalse.Emitter.BrFalse(0);

        Visit(@if.Then);
        var br = BeginTextSnippet(@if);
        br.Emitter.Br(0);

        Visit(@if.Else);

        // because visit param is before VisitLeaf, we can't use ref of elseSnippet.Symbol to jump.
        // snippet structure: | ... | else param1 | else param2 | ... | elseSnippet |
        AddSymbolRef(brFalse, br.EndSymbol, -4, 4, true, 1);
        var endSnippet = BeginTextSnippet(@if);
        AddSymbolRef(br, endSnippet.BeginSymbol, -4, 4, true, 1);
        return endSnippet;
    }

    private TextSnippet Visit(TensorConst expr, Tensor tensor)
    {
        var buffer = WriteRdata(tensor.BytesBuffer, _alignment);

        // stack: dtype shape strides buffer
        var snippet = BeginTextSnippet(expr);
        LeaGp(_rdataGpid, buffer);
        LdStrides(tensor.Strides);
        LdShape(tensor.Dimensions);
        LdDataType(tensor.ElementType);
        Emitter.LdTensor();
        return snippet;
    }

    private Symbol WriteRdata(DataType dataType)
    {
        if (!_context.DataTypes.TryGetValue(dataType, out var symbol))
        {
            symbol = AddSymbol(WellknownSectionNames.Rdata);

            TypeSerializer.Serialize(_context.RdataWriter, dataType);
            _context.DataTypes.Add(dataType, symbol);
        }

        return symbol;
    }

    private Symbol WriteRdata(ReadOnlySpan<byte> data, int alignment)
    {
        _context.RdataWriter.AlignPosition(alignment);
        var symbol = AddSymbol(WellknownSectionNames.Rdata);
        _context.RdataWriter.Write(data);
        return symbol;
    }

    private Symbol AddSymbol(string sectionName)
    {
        var position = sectionName == WellknownSectionNames.Text ? 0 : _context.RdataWriter.Position();
        return new Symbol(sectionName, position);
    }

    private SymbolRef AddSymbolRef(Symbol symbol, int positionOffset, int length, bool relative = false, int offset = 0)
    {
        return AddSymbolRef(CurrentTextSnippet, symbol, positionOffset, length, relative, offset);
    }

    private SymbolRef AddSymbolRef(TextSnippet snippet, Symbol symbol, int positionOffset, int length, bool relative = false, int offset = 0)
    {
        var symbolRef = new SymbolRef(snippet.Emitter.Position + positionOffset, length, symbol, relative, offset);
        snippet.SymbolRefs.Add(symbolRef);
        return symbolRef;
    }

    private FunctionRef AddFunctionRef(BaseFunction callable, FunctionIdComponent component, int positionOffset, int length, int offset = 0)
    {
        var functionRef = new FunctionRef(Emitter.Position + positionOffset, length, callable, component, offset);
        CurrentTextSnippet.FunctionRefs.Add(functionRef);
        return functionRef;
    }

    private void LeaGp(byte gpid, Symbol symbol, int offset = 0)
    {
        AddSymbolRef(symbol, 2, 4, false, offset);
        Emitter.LeaGP(gpid, 0);
    }

    private void LdFunctionId(BaseFunction callable)
    {
        AddFunctionRef(callable, FunctionIdComponent.FunctionId, 1, 4, 0);
        Emitter.LdcI4(0);
        AddFunctionRef(callable, FunctionIdComponent.ModuleId, 1, 4, 0);
        Emitter.LdcI4(0);
    }

    private void LdShape(ReadOnlySpan<int> shape)
    {
        for (int i = shape.Length - 1; i >= 0; i--)
        {
            Emitter.LdcI4(shape[i]);
        }

        Emitter.LdcI4(shape.Length);
    }

    private void LdStrides(ReadOnlySpan<int> strides)
    {
        for (int i = strides.Length - 1; i >= 0; i--)
        {
            Emitter.LdcI4(strides[i]);
        }

        Emitter.LdcI4(strides.Length);
    }

    private void LdDataType(DataType dataType)
    {
        var dtype = WriteRdata(dataType);
        LeaGp(_rdataGpid, dtype);
        Emitter.LdDataType();
    }

    private TextSnippet BeginTextSnippet(Expr expr)
    {
        var snippet = new TextSnippet(
            expr,
            AddSymbol(WellknownSectionNames.Text),
            AddSymbol(WellknownSectionNames.Text));
        _currentTextSnippet = snippet;
        _context.AddTextSnippet(snippet);
        return snippet;
    }
}
