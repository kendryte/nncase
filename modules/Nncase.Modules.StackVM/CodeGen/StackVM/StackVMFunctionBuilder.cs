// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.Tensors;
using Nncase.Runtime.StackVM;
using Nncase.Utilities;
using static Nncase.CodeGen.CodeGenDumper;

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
        foreach (var basicBlock in _context.BasicBlocks)
        {
            foreach (var snippet in basicBlock.TextSnippets)
            {
                snippet.RefCount = snippet.UseCount;
            }
        }

        var localSet = new HashSet<int>();
        var sourceMap = new List<(Expr, (long, long))>();
        Var Tag(string name) => new(name, AnyType.Default);
        int i = 1;
        foreach (var basicBlock in _context.BasicBlocks)
        {
            sourceMap.Add((Tag("Begin"), (0, 0)));
            foreach (var snippet in basicBlock.TextSnippets)
            {
                if (snippet.Expr is Call { Target: Reshape } snippetCall && snippetCall.Arguments[0] is Call gatherCall && gatherCall.Target is Gather)
                {
                    i++;
                }
                SymbolAddrs.Add(snippet.BeginSymbol, _textEmitter.Position);
                var begin = _textEmitter.Position;

                // end of if
                if (basicBlock.Prev.Count > 1 && _context.AllocInfo.TryGetValue(snippet, out var uses))
                {
                    sourceMap.Add((Tag("endOfIf"), (0, 0)));
                    foreach (var inputSnippet in uses)
                    {
                        var localId = _snippetLocals[inputSnippet];
                        RefCountReduce(inputSnippet, localId);

                        if (inputSnippet.RefCount == 0)
                        {
                            _snippetLocals.Remove(inputSnippet);
                            localSet.Remove(localId);
                            sourceMap.Add((Tag($"release {ToStr(inputSnippet.Expr)}"), (0, 0)));
                        }
                    }

                    Debug.Assert(snippet.InputSnippets.Count == 0, "snippet end of if is should be 0 input");
                }

                // 2.1 Load inputs
                foreach (var inputSnippet in snippet.InputSnippets)
                {
                    // in locals
                    if (inputSnippet.OutputInLocal)
                    {
                        var localId = _snippetLocals[inputSnippet];
                        _textEmitter.Ldlocal(localId);

                        if (CodegenUtility.NormalReduceCount(snippet, inputSnippet))
                        {
                            var beforePos = _textEmitter.Position;
                            RefCountReduce(inputSnippet, localId);
                            if (inputSnippet.RefCount == 0)
                            {
                                localSet.Remove(localId);
                                sourceMap.Add((Tag($"release {ToStr(inputSnippet.Expr)}"), (beforePos, _textEmitter.Position)));
                                _snippetLocals.Remove(inputSnippet);
                            }
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
                    localSet.Add(localId);
                    _snippetLocals.Add(snippet, localId);
                    _textEmitter.Stlocal(localId);
                }

                SymbolAddrs.Add(snippet.EndSymbol, _textEmitter.Position);
                var end = _textEmitter.Position;
                sourceMap.Add((snippet.Expr, (begin, end)));
            }
            sourceMap.Add((Tag("End"), (0, 0)));
        }

        if (DumpScope.Current.IsEnabled(DumpFlags.CodeGen))
        {
            WriteDebugInfo(Id, 0, sourceMap);
        }
    }

    private void RefCountReduce(TextSnippet inputSnippet, ushort localId)
    {
        // last usage, set the local to null
        if (--inputSnippet.RefCount == 0)
        {
            _textEmitter.LdNull();
            _textEmitter.Stlocal(localId);
            _localsAllocator.Free(localId);
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
