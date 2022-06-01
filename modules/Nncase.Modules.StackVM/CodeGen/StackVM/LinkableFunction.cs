using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.CodeGen.StackVM;

internal class LinkableFunction
{
    public LinkableFunction(uint id, Function sourceFunction, IEnumerable<FunctionRef> functionRefs, ushort maxLocals, byte[] text)
    {
        Id = id;
        SourceFunction = sourceFunction;
        FunctionRefs = functionRefs;
        MaxLocals = maxLocals;
        Text = text;
    }

    public uint Id { get; }

    public Function SourceFunction { get; }

    public IEnumerable<FunctionRef> FunctionRefs { get; }

    public ushort MaxLocals { get; }

    public byte[] Text { get; }
}
