// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.Runtime.StackVM;

namespace Nncase.CodeGen.StackVM;

/// <summary>
/// StackVM function builder.
/// </summary>
internal class StackVMFunctionBuilder
{
    private readonly uint _id;
    private readonly CodeGenContext _context;
    private readonly Dictionary<Symbol, long> _symbolAddrs = new Dictionary<Symbol, long>();
    private readonly MemoryStream _textContent = new MemoryStream();
    private readonly BinaryWriter _textWriter;
    private readonly StackVMEmitter _textEmitter;
    private readonly LocalsAllocator _localsAllocator = new LocalsAllocator();
    private readonly Dictionary<TextSnippet, ushort> _snippetLocals = new Dictionary<TextSnippet, ushort>();
    private readonly List<SymbolRef> _symbolRefs = new List<SymbolRef>();
    private readonly List<FunctionRef> _functionRefs = new List<FunctionRef>();

    public StackVMFunctionBuilder(uint id, BinaryWriter rdataWriter)
    {
        _id = id;
        _context = new CodeGenContext(rdataWriter);
        _textWriter = new BinaryWriter(_textContent, Encoding.UTF8, leaveOpen: true);
        _textEmitter = new StackVMEmitter(_textWriter);
    }

    public LinkableFunction Build(Function function)
    {
        // 1. Compile
        Compile(function);

        // 2. Write text
        WriteText();

        // 3. Fix addrs
        FixAddrs();
        return new LinkableFunction(_id, function, _functionRefs, _localsAllocator.MaxCount, _textContent.ToArray());
    }

    private void Compile(Function function)
    {
        new CodeGenVisitor(function, _context).Visit(function.Body);
    }

    private void WriteText()
    {
        // 1. Assign ref counts
        foreach (var snippet in _context.TextSnippets)
        {
            snippet.RefCount = snippet.UseCount;
        }

        // 2. Gen code
        foreach (var snippet in _context.TextSnippets)
        {
            _symbolAddrs.Add(snippet.Symbol, _textEmitter.Position);

            // 2.1 Load inputs
            foreach (var inputSnippet in snippet.InputSnippets)
            {
                // in locals
                if (inputSnippet.UseCount > 1)
                {
                    var localId = _snippetLocals[inputSnippet];
                    _textEmitter.Ldlocal(localId);

                    // last usage, set the local to null
                    if (--inputSnippet.RefCount == 0)
                    {
                        _textEmitter.LdNull();
                        _textEmitter.Stlocal(localId);
                        _localsAllocator.Free(localId);
                    }
                }
            }

            // 2.2 Write body
            var bodyPosition = _textEmitter.Position;
            foreach (var refer in snippet.SymbolRefs)
            {
                _symbolRefs.Add(refer with { Position = refer.Position + bodyPosition });
            }

            foreach (var refer in snippet.FunctionRefs)
            {
                _functionRefs.Add(refer with { Position = refer.Position + bodyPosition });
            }

            snippet.Writer.Flush();
            _textWriter.Write(snippet.Text.ToArray());

            // 2.3 Store output
            // in locals
            if (snippet.UseCount > 1)
            {
                var localId = _localsAllocator.Allocate();
                _snippetLocals.Add(snippet, localId);
                _textEmitter.Stlocal(localId);
            }
        }

        _textWriter.Flush();
    }

    private void FixAddrs()
    {
        foreach (var refer in _symbolRefs)
        {
            _textWriter.Position(refer.Position);
            var symbolAddr = refer.Symbol.Position;
            if (refer.Symbol.Section == SectionKind.Text)
            {
                symbolAddr += _symbolAddrs[refer.Symbol];
            }

            _textWriter.WriteByLength(symbolAddr + refer.Offset, refer.Length);
        }
    }

    private class LocalsAllocator
    {
        private SortedSet<ushort> _locals = new SortedSet<ushort>();

        public ushort MaxCount { get; private set; }

        public ushort Allocate()
        {
            if (_locals.Count == 0)
            {
                var id = MaxCount++;
                return id;
            }
            else
            {
                var id = _locals.Min;
                _locals.Remove(id);
                return id;
            }
        }

        public void Free(ushort id)
        {
            if (!_locals.Add(id))
            {
                throw new InvalidOperationException();
            }
        }
    }
}
