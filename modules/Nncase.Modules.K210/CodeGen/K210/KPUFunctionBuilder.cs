// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.CodeGen.StackVM;
using Nncase.IR;
using Nncase.Runtime.K210;

namespace Nncase.CodeGen.K210;

/// <summary>
/// K210 function builder.
/// </summary>
internal class KPUFunctionBuilder : FunctionBuilder
{
    private readonly CodeGenContext _context;
    private readonly Emitter _textEmitter;
    private readonly LocalsAllocator _localsAllocator;
    private readonly Dictionary<TextSnippet, ushort> _snippetLocals;
    private readonly SectionManager _rdataWriter;

    public KPUFunctionBuilder(uint id, SectionManager sectionManager)
        : base(id, sectionManager)
    {
        _context = new CodeGenContext(sectionManager.GetWriter(WellknownSectionNames.Rdata));
        _textEmitter = new Emitter(TextWriter);
    }

    public ITarget Target { get; }

    protected override void Compile(Callable callable)
    {
        var function = (Function)callable;
        new CodeGenVisitor(function, _context).Visit(function.Body);
        
        /*var lower_module = new IRModule();
        var functions = (Function) callable;

        foreach (Callable f in functions)
        {
            lower_module.Add(f);
        }
        
        var passManager = new Transform.PassManager(lower_module,
            new Transform.RunPassOptions(Target))
        {
            new Transform.TIRPass("BufferStage")
            {
                Transform.ExtMutator.RemoveBufferDecl(), // 删除内部的T.buffer
                Transform.Mutator.RemoveNop(), // 删除内部的T.Nop
                Transform.Mutator.UnRollLoop(), // 展开循环.
                Transform.ExtMutator.FoldConstCall(), // 常量折叠所有的指令参数.
                Transform.Mutator.FoldConstTuple(), // 合成consttuple
                Transform.Mutator.FoldLet(), // 折叠let表达式.
                Transform.Mutator.FoldIfThen(), // 折叠let表达式.
                Transform.Mutator.FlattenSequential(), // 折叠sequential.
            },
            new Transform.TIRPass("MiddleStage")
            {
                //Transform.ExtMutator.FoldConstCall(), // 常量折叠所有的指令参数.
                Transform.Mutator.FoldConstTuple(), // 合成consttuple
                //Transform.ExtMutator.FoldDDrBuffer(new()), // 分配ddr of buffer
                //Transform.ExtMutator.FoldGlbBuffer(), // 分配glb mmuof buffer
                Transform.Mutator.UnRollLoop(), // 展开循环.
                Transform.Mutator.FoldLet(), // 折叠let表达式.
                Transform.Mutator.FlattenSequential(), // 折叠sequential.
                //Transform.ExtMutator.UnFoldCustomOp(), // 展开自定义的op.
                Transform.Mutator.RemoveNop(), // 删除内部的T.Nop
            },
            new Transform.TIRPass("InstStage")
            {
                //Transform.ExtMutator.FoldCustomOp() // 折叠自定义op.
            },
        };
        passManager.RunAsync().Wait();
        lower_module.Callables.OfType<IR.Function>().Select((f, i) => new KPUFunctionBuilder((uint)i, _rdataWriter).Build(f)).ToArray();*/
    }


protected override ILinkableFunction CreateLinkableFunction(uint id, Callable callable,
        IReadOnlyList<FunctionRef> functionRefs, byte[] text)
    {
        return new KPULinkableFunction(id, (Function) callable, functionRefs, text);
    }

    protected override void WriteText()
    {
        foreach (var snippet in _context.TextSnippets)
        {
            snippet.RefCount = snippet.UseCount;
        }

        foreach (var snippet in _context.TextSnippets)
        {
            SymbolAddrs.Add(snippet.Symbol, _textEmitter.Position);

            foreach (var inputSnippet in snippet.InputSnippets)
            {
                if (inputSnippet.OutputInLocal)
                {
                    var localId = _snippetLocals[inputSnippet];
                    _textEmitter.Ldlocal(localId);

                    if (--inputSnippet.RefCount == 0)
                    {
                        _textEmitter.LdNull();
                        _textEmitter.Stlocal(localId);
                        _localsAllocator.Free(localId);
                    }
                }
            }

            var bodyPosition = _textEmitter.Position;
            foreach (var refer in snippet.SymbolRefs)
            {
                SymbolRefs.Add(refer with {Position = refer.Position + bodyPosition});
            }

            foreach (var refer in snippet.FunctionRefs)
            {
                FunctionRefs.Add(refer with {Position = refer.Position + bodyPosition});
            }

            snippet.Writer.Flush();
            TextWriter.Write(snippet.Text.ToArray());

            if (snippet.OutputInLocal)
            {
                var localId = _localsAllocator.Allocate();
                _snippetLocals.Add(snippet, localId);
                _textEmitter.Stlocal(localId);
            }
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
