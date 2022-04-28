// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.CodeGen.StackVM;

public enum SectionKind
{
    Text,
    Rdata
}

public class Symbol
{
    public SectionKind Section { get; set; }

    public int Index { get; set; }
}

public class SymbolRef
{
    public long Position { get; set; }

    public int Length { get; set; }

    public Symbol Symbol { get; set; }

    public int Offset { get; set; }
}

internal class TextSnippet
{
    public TextSnippet(Symbol symbol)
    {
        Symbol = symbol;
        Writer = new BinaryWriter(Text, Encoding.UTF8, leaveOpen: true);
        Emitter = new StackVMEmitter(Writer);
    }

    public MemoryStream Text { get; } = new MemoryStream();

    public BinaryWriter Writer { get; }

    public StackVMEmitter Emitter { get; }

    public List<SymbolRef> SymbolRefs { get; } = new List<SymbolRef>();

    public List<TextSnippet> InputSnippets { get; } = new List<TextSnippet>();

    public Symbol Symbol { get; }
}

internal partial class CodeGenVisitor : ExprVisitor<TextSnippet, IRType>
{
    private const int _alignment = 8;
    private const byte _rdataGpid = 0;

    private readonly Function _function;

    public CodeGenVisitor(Function function)
    {
        _function = function;
        _rdataWriter = new BinaryWriter(_rdataContent);
    }

    public override TextSnippet VisitLeaf(Const expr)
    {
        if (expr is TensorConst tc)
        {
            return Visit(tc.Value);
        }
        else
        {
            return Visit(new IR.Tuple(((TupleConst)expr).Fields));
        }
    }

    public override TextSnippet VisitLeaf(Var expr)
    {
        var snippet = BeginTextSnippet();
        Emitter.Ldarg((uint)_function.Parameters.IndexOf(expr));
        return snippet;
    }

    public override TextSnippet VisitLeaf(IR.Tuple expr)
    {
        var snippet = BeginTextSnippet();
        foreach (var field in expr.Fields.Reverse())
        {
            snippet.InputSnippets.Add(Visit(field));
        }

        Emitter.LdTuple();
        return snippet;
    }

    public override TextSnippet Visit(Function expr)
    {
        throw new NotSupportedException();
    }

    public override TextSnippet VisitLeaf(Op expr)
    {
        return null!;
    }

    public override TextSnippet VisitLeaf(Call expr)
    {
        if (expr.Target is Op op)
        {
            var snippet = BeginTextSnippet();
            foreach (var param in expr.Parameters.Reverse())
            {
                snippet.InputSnippets.Add(Visit(param));
            }

            EmitTensorCall(op);
            return snippet;
        }
        else
        {
            throw new NotSupportedException();
        }
    }

    private TextSnippet Visit(Tensor tensor)
    {
        var buffer = WriteRdata(tensor.BytesBuffer, _alignment);

        // stack: dtype shape strides buffer
        var snippet = BeginTextSnippet();
        LeaGp(_rdataGpid, buffer);
        LdStrides(tensor.Strides);
        LdShape(tensor.Dimensions);
        LdDataType(tensor.ElementType);
        Emitter.LdTensor();
        return snippet;
    }

    private Symbol WriteRdata(DataType dataType)
    {
        if (!_dataTypes.TryGetValue(dataType, out var symbol))
        {
            symbol = AddSymbol(SectionKind.Rdata);

            DataTypeSerializer.Serialize(_rdataWriter, dataType);
            _dataTypes.Add(dataType, symbol);
        }

        return symbol;
    }

    private Symbol WriteRdata(ReadOnlySpan<byte> data, int alignment)
    {
        _rdataWriter.AlignPosition(alignment);
        var symbol = AddSymbol(SectionKind.Rdata);
        _rdataWriter.Write(data);
        return symbol;
    }

    private Symbol AddSymbol(SectionKind kind)
    {
        return new Symbol { Section = kind };
    }

    private SymbolRef AddSymbolRef(Symbol symbol, int positionOffset, int length, int offset = 0)
    {
        var symbolRef = new SymbolRef
        {
            Position = Emitter.Position + positionOffset,
            Symbol = symbol,
            Length = length,
            Offset = offset,
        };
        CurrentTextSnippet.SymbolRefs.Add(symbolRef);
        return symbolRef;
    }

    private void LeaGp(byte gpid, Symbol symbol, int offset = 0)
    {
        AddSymbolRef(symbol, 2, 1, offset);
        Emitter.LeaGP(gpid, 0);
    }

    private void LdShape(ReadOnlySpan<int> shape)
    {
        foreach (var dim in shape)
        {
            Emitter.LdcI4(dim);
        }

        Emitter.LdcI4(shape.Length);
        Emitter.LdShape();
    }

    private void LdStrides(ReadOnlySpan<int> strides)
    {
        foreach (var dim in strides)
        {
            Emitter.LdcI4(dim);
        }

        Emitter.LdcI4(strides.Length);
        Emitter.LdStrides();
    }

    private void LdDataType(DataType dataType)
    {
        var dtype = WriteRdata(dataType);
        LeaGp(_rdataGpid, dtype);
        Emitter.LdDataType();
    }

    private TextSnippet BeginTextSnippet()
    {
        var snippet = new TextSnippet(AddSymbol(SectionKind.Text));
        _currentTextSnippet = snippet;
        return snippet;
    }

    private TextSnippet? _currentTextSnippet;

    private TextSnippet CurrentTextSnippet => _currentTextSnippet;

    private StackVMEmitter Emitter => CurrentTextSnippet.Emitter;

    private readonly List<TextSnippet> _textSnippets = new List<TextSnippet>();
    private readonly MemoryStream _rdataContent = new MemoryStream();
    private readonly BinaryWriter _rdataWriter;
    private readonly Dictionary<DataType, Symbol> _dataTypes = new Dictionary<DataType, Symbol>();
}
