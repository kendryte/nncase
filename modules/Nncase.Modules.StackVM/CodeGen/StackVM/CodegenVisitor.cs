// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.CodeGen.StackVM;

internal class CodegenVisitor : ExprVisitor<Expr, Symbol>
{
    public CodegenVisitor()
    {
        _rdataWriter = new BinaryWriter(_rdataContent);
    }

    public override Expr VisitLeaf(Const expr)
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

    private Expr Visit(Tensor tensor)
    {
        var dtype = WriteRdata(tensor.ElementType);

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

    private Symbol AddSymbol(SectionKind kind)
    {

    }

    private Symbol LeaGp(int gpid, SymbolRef symbol)
    {

    }

    private readonly List<byte[]> _textContent = new List<byte[]>();
    private readonly MemoryStream _rdataContent = new MemoryStream();
    private readonly BinaryWriter _rdataWriter;
    private readonly Dictionary<DataType, Symbol> _dataTypes = new Dictionary<DataType, Symbol>();

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
        public Symbol Symbol { get; set; }

        public int Offset { get; set; }
    }
}
