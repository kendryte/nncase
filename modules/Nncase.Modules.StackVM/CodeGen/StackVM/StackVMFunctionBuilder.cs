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
internal class StackVMFunctionBuilder : FunctionBuilder
{
    private readonly CodeGenContext _context;
    private readonly StackVMEmitter _textEmitter;
    private readonly LocalsAllocator _localsAllocator = new LocalsAllocator();
    private readonly Dictionary<TextSnippet, ushort> _snippetLocals = new Dictionary<TextSnippet, ushort>();

    public StackVMFunctionBuilder(uint id, SectionManager sectionManager)
        : base(id, sectionManager)
    {
        _context = new CodeGenContext(sectionManager.GetWriter(WellknownSectionNames.Rdata));
        _textEmitter = new StackVMEmitter(TextWriter);
    }

    protected override ILinkableFunction CreateLinkableFunction(uint id, BaseFunction callable, IReadOnlyList<FunctionRef> functionRefs, byte[] text)
    {
        return new StackVMLinkableFunction(id, callable, functionRefs, _localsAllocator.MaxCount, text, _context.CustomCallModules);
    }

    protected override void Compile(BaseFunction callable)
    {
        new CodeGenVisitor(callable, _context).Visit(callable);
    }

    protected override void WriteText()
    {
        // 1. Assign ref counts
        foreach (var snippet in _context.TextSnippets)
        {
            snippet.RefCount = snippet.UseCount;
        }

        // 2. Gen code
        foreach (var snippet in _context.TextSnippets)
        {
            SymbolAddrs.Add(snippet.BeginSymbol, _textEmitter.Position);

            // 2.1 Load inputs
            foreach (var inputSnippet in snippet.InputSnippets)
            {
                // in locals
                if (inputSnippet.OutputInLocal)
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
                SymbolRefs.Add(refer with { Position = refer.Position + bodyPosition });
            }

            foreach (var refer in snippet.FunctionRefs)
            {
                FunctionRefs.Add(refer with { Position = refer.Position + bodyPosition });
            }

            snippet.Writer.Flush();
            TextWriter.Write(snippet.Text.ToArray());

            // 2.3 Store output
            // in locals
            if (snippet.OutputInLocal)
            {
                var localId = _localsAllocator.Allocate();
                _snippetLocals.Add(snippet, localId);
                _textEmitter.Stlocal(localId);
            }

            SymbolAddrs.Add(snippet.EndSymbol, _textEmitter.Position);
        }
    }

    private class LocalsAllocator
    {
        private readonly SortedSet<ushort> _locals = new SortedSet<ushort>();

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
