﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Numerics;
using System.Reactive;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Tensors;

namespace Nncase.CodeGen.StackVM;

internal class TextSnippet
{
    private readonly List<TextSnippet> _inputSnippets = new List<TextSnippet>();

    public TextSnippet(Expr expr, Symbol beginSymbol, Symbol endSymbol, BasicBlock basicBlock)
    {
        Expr = expr;
        BeginSymbol = beginSymbol;
        EndSymbol = endSymbol;
        Writer = new BinaryWriter(Text, Encoding.UTF8, leaveOpen: true);
        Emitter = new StackVMEmitter(Writer);
        BasicBlock = basicBlock;
    }

    public BasicBlock BasicBlock { get; set; }

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

    public void AddInput(TextSnippet input, bool addCount)
    {
        _inputSnippets.Add(input);
        if (addCount)
        {
            input.UseCount++;
        }
    }

    public void AddUseCount()
    {
        UseCount++;
    }
}

internal class BasicBlock
{
    private static int _count;

    private readonly List<TextSnippet> _textSnippets = new();

    public BasicBlock()
    {
        Id = _count++;
    }

    public BasicBlock(BasicBlock[] prevs)
    {
        Id = _count++;
        Prev = Prev.Concat(prevs).ToList();
        foreach (var prev in prevs)
        {
            prev.Nexts.Add(this);
        }
    }

    public List<BasicBlock> Prev { get; set; } = new();

    public List<BasicBlock> Nexts { get; set; } = new();

    public int Id { get; set; }

    public IReadOnlyList<TextSnippet> TextSnippets => _textSnippets;

    public void AddNext(BasicBlock next)
    {
        next.Prev.Add(this);
        Nexts.Add(next);
    }

    public void AddTextSnippet(TextSnippet textSnippet)
    {
        _textSnippets.Add(textSnippet);
    }
}

internal class CodeGenContext
{
    private readonly List<BasicBlock> _basicBlocks = new();
    private readonly HashSet<ModuleType> _custom_call_modules = new();

    public CodeGenContext(BinaryWriter rdataWriter, Dictionary<TensorConst, Symbol> constSymbols)
    {
        RdataWriter = rdataWriter;
        ConstSymbols = constSymbols;
    }

    public Dictionary<TextSnippet, HashSet<TextSnippet>> AllocInfo { get; set; } = new();

    public BinaryWriter RdataWriter { get; }

    public Dictionary<DataType, Symbol> DataTypes { get; } = new Dictionary<DataType, Symbol>();

    public IReadOnlyList<BasicBlock> BasicBlocks => _basicBlocks;

    public IReadOnlySet<ModuleType> CustomCallModules => _custom_call_modules;

    public Dictionary<TensorConst, Symbol> ConstSymbols { get; }

    public void AddBasicBlock(BasicBlock basicBlock)
    {
        _basicBlocks.Add(basicBlock);
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
    private readonly List<TextSnippet> _refTextSnippets = new();

    private TextSnippet? _currentTextSnippet;
    private readonly BasicBlock? _currentBasicBlock;

    public CodeGenVisitor(BaseFunction function, CodeGenContext context)
    {
        _function = function;
        _context = context;
        var basicBlock = new BasicBlock();
        _context.AddBasicBlock(basicBlock);
        _currentBasicBlock = basicBlock;
    }

    private TextSnippet CurrentTextSnippet => _currentTextSnippet!;

    private BasicBlock CurrentBasicBlock => _currentBasicBlock!;

    private StackVMEmitter Emitter => CurrentTextSnippet.Emitter;

    public (BasicBlock BB, List<TextSnippet> SnippetSet) SubBlock(Expr expr)
    {
        var visitor = new CodeGenVisitor(_function, _context);
        var subBlockFirst = visitor.CurrentBasicBlock;

        CurrentBasicBlock.AddNext(subBlockFirst);
        foreach (var (key, value) in ExprMemo)
        {
            visitor.ExprMemo[key] = value;
        }

        visitor.Visit(expr);
        var refTextSnippets = visitor._refTextSnippets;
        var subBlockEnd = visitor.CurrentBasicBlock;
        return (subBlockEnd, refTextSnippets);
    }

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
            snippet.AddInput(inputSnippet, true);
        }

        Emitter.LdcI4(expr.Count);
        Emitter.LdTuple();
        return snippet;
    }

    protected override TextSnippet VisitFunction(Function expr)
    {
        if (ReferenceEquals(expr, _function))
        {
            var bodySnippet = Visit(expr.Body);
            bodySnippet.Emitter.Ret();
            return bodySnippet;
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
            if (CodegenUtility.NormalReduceCount(paramSnippet, snippet))
            {
                snippet.AddInput(paramSnippet, true);
            }
            else
            {
                snippet.AddInput(paramSnippet, false);
                _refTextSnippets.Add(paramSnippet);
            }

            paramSnippet.MaxUserParameters = Math.Max(paramSnippet.MaxUserParameters, expr.Arguments.Length);
        }

        EmitCallable(expr.Target, expr.Arguments.Length);
        return snippet;
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
    /// <param name="expr">If expr.</param>
    /// <returns>TextSnippet.</returns>
    protected override TextSnippet VisitLeafIf(If expr)
    {
        var condSnippet = Visit(expr.Condition);

        var snippet = BeginTextSnippet(expr);
        foreach (var param in expr.Arguments.ToArray().Reverse())
        {
            var paramSnippet = Visit(param);
            if (CodegenUtility.NormalReduceCount(paramSnippet, snippet))
            {
                snippet.AddInput(paramSnippet, true);
            }
            else
            {
                snippet.AddInput(paramSnippet, false);
                _refTextSnippets.Add(paramSnippet);
            }

            paramSnippet.MaxUserParameters = Math.Max(paramSnippet.MaxUserParameters, expr.Arguments.Length + 1);
        }

        snippet.AddInput(condSnippet, true);
        condSnippet.MaxUserParameters = Math.Max(condSnippet.MaxUserParameters, expr.Arguments.Length + 1);
        Emitter.LdScalar();

        var brFalsePos = Emitter.Position;
        Emitter.BrFalse(0);
        EmitCallable(expr.Then, expr.Arguments.Length);
        var brPos = Emitter.Position;
        Emitter.Br(0);

        var elsePos = Emitter.Position;
        EmitCallable(expr.Else, expr.Arguments.Length);
        var endPos = Emitter.Position;

        // fixup
        Emitter.Position = brFalsePos;
        Emitter.BrFalse((int)(elsePos - brFalsePos));

        Emitter.Position = brPos;
        Emitter.Br((int)(endPos - brPos));
        Emitter.Position = endPos;
        return snippet;
    }

    private void EmitCallable(Expr callable, int argumentsCount)
    {
        if (callable is CustomOp custom_op)
        {
            _context.AddCustomCallModule(custom_op.ModuleType);
            Emitter.CusCall(custom_op.RegisteredName, custom_op.SerializeFields(), checked((ushort)argumentsCount));
        }
        else if (callable is Op op)
        {
            EmitTensorCall(op);
        }
        else if (callable is PrimFunctionWrapper wrapper)
        {
            LdFunctionId(wrapper.Target);
            Emitter.ExtCall(checked((ushort)wrapper.ParameterTypes.Count()), true);
        }
        else if (callable is Function func)
        {
            LdFunctionId(func);
            Emitter.ExtCall(checked((ushort)func.Parameters.Length), false);
        }
        else
        {
            throw new NotSupportedException(callable.GetType().Name);
        }
    }

    private void MergeSnippetSet(List<TextSnippet> thenSet, List<TextSnippet> elseSet, TextSnippet endSnippet)
    {
        var useSnippetSet = thenSet.Concat(elseSet).ToHashSet();
        foreach (var snippet in useSnippetSet)
        {
            snippet.AddUseCount();
        }

        _context.AllocInfo[endSnippet] = useSnippetSet;
    }

    private TextSnippet Visit(TensorConst expr, Tensor tensor)
    {
        if (!_context.ConstSymbols.TryGetValue(expr, out var buffer))
        {
            buffer = WriteRdata(tensor, _alignment);
            _context.ConstSymbols.Add(expr, buffer);
        }

        // stack: dtype shape strides buffer
        var snippet = BeginTextSnippet(expr);
        LeaGp(_rdataGpid, buffer);
        LdStrides(tensor.Strides.ToInts());
        LdShape(tensor.Dimensions.ToInts());
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

    private Symbol WriteRdata(Tensor tensor, int alignment)
    {
        _context.RdataWriter.AlignPosition(alignment);
        var symbol = AddSymbol(WellknownSectionNames.Rdata);
        tensor.Serialize(_context.RdataWriter.BaseStream);
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
        AddSymbolRef(symbol, 2, 8, false, offset);
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
            AddSymbol(WellknownSectionNames.Text),
            CurrentBasicBlock);
        _currentTextSnippet = snippet;
        CurrentBasicBlock.AddTextSnippet(snippet);
        return snippet;
    }
}
