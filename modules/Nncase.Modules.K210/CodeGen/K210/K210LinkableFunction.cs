using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.CodeGen.K210;

internal class K210LinkableFunction : ILinkableFunction
{
    public K210LinkableFunction(uint id, Function sourceFunction, IEnumerable<FunctionRef> functionRefs, byte[] text)
    {
        Id = id;
        SourceFunction = sourceFunction;
        Text = text;
        FunctionRefs = functionRefs;
    }

    public uint Id { get; }

    public Function SourceFunction { get; }

    public byte[] Text { get; }

    public IEnumerable<FunctionRef> FunctionRefs { get; }

    public IReadOnlyList<ILinkedSection> Sections => Array.Empty<ILinkedSection>();
}
