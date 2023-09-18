// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.CodeGen;

/// <summary>
/// Function builder.
/// </summary>
public abstract class FunctionBuilder : IDisposable
{
    private readonly MemoryStream _textContent = new MemoryStream();

    public FunctionBuilder(uint id, SectionManager sectionManager)
    {
        Id = id;
        TextWriter = new BinaryWriter(_textContent, Encoding.UTF8, leaveOpen: true);
        SectionManager = sectionManager;
    }

    public uint Id { get; }

    public BinaryWriter TextWriter { get; }

    public SectionManager SectionManager { get; }

    protected Dictionary<Symbol, long> SymbolAddrs { get; } = new Dictionary<Symbol, long>();

    protected List<SymbolRef> SymbolRefs { get; } = new List<SymbolRef>();

    protected List<FunctionRef> FunctionRefs { get; } = new List<FunctionRef>();

    public ILinkableFunction Build(BaseFunction callable)
    {
        // 1. Compile
        Compile(callable);

        // 2. Write text
        WriteText();

        // 3. Fix addrs
        FixAddrs();
        TextWriter.Flush();
        return CreateLinkableFunction(Id, callable, FunctionRefs, _textContent);
    }

    public void Dispose()
    {
        throw new NotImplementedException();
    }

    protected abstract void Compile(BaseFunction callable);

    protected abstract void WriteText();

    protected abstract ILinkableFunction CreateLinkableFunction(uint id, BaseFunction callable, IReadOnlyList<FunctionRef> functionRefs, Stream text);

    private void FixAddrs()
    {
        foreach (var refer in SymbolRefs)
        {
            TextWriter.Position(refer.Position);
            var symbolAddr = refer.Symbol.Position;
            if (refer.Symbol.Section == WellknownSectionNames.Text)
            {
                symbolAddr += SymbolAddrs[refer.Symbol];
            }

            long originValue = symbolAddr + refer.Offset;
            long value = refer.Relative ? originValue - refer.Position : originValue;

            // todo: neg addr is error
            TextWriter.WriteByLength(value, refer.Length);
        }
    }
}
