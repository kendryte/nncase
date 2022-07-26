using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.CodeGen;

public interface ILinkableFunction
{
    public uint Id { get; }

    public BaseFunction SourceFunction { get; }

    public IEnumerable<FunctionRef> FunctionRefs { get; }

    public byte[] Text { get; }

    public IReadOnlyList<ILinkedSection> Sections { get; }
}
