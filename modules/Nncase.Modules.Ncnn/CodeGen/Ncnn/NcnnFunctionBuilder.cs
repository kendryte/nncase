// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Reactive;
using System.Text;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.Ncnn;
using Nncase.IR.Tensors;
using Nncase.Utilities;
using static Nncase.CodeGen.CodeGenDumper;

namespace Nncase.CodeGen.Ncnn;

/// <summary>
/// Ncnn function builder.
/// </summary>
internal class NcnnFunctionBuilder : FunctionBuilder
{
    private readonly NcnnEmitter _emitter;
    private string[]? _inputs;
    private string[]? _outputs;

    public NcnnFunctionBuilder(uint id, SectionManager sectionManager)
        : base(id, sectionManager)
    {
        _emitter = new NcnnEmitter(sectionManager.GetWriter(WellknownSectionNames.Rdata));
    }

    protected override ILinkableFunction CreateLinkableFunction(uint id, BaseFunction callable, IReadOnlyList<FunctionRef> functionRefs, Stream text)
    {
        return new NcnnLinkableFunction(id, callable, functionRefs, text, _inputs!, _outputs!);
    }

    protected override void Compile(BaseFunction callable)
    {
        var visitor = new CodeGenVisitor(_emitter);
        _outputs = visitor.Visit(callable).Split(',');
        _inputs = visitor.Inputs.ToArray();
    }

    protected override void WriteText()
    {
        _emitter.SaveParam(TextWriter.BaseStream);
    }

    private class CodeGenVisitor : ExprVisitor<string, Unit>
    {
        private readonly NcnnEmitter _emitter;

        private int _layerId;

        public CodeGenVisitor(NcnnEmitter emitter)
        {
            _emitter = emitter;
        }

        public List<string> Inputs { get; } = new();

        protected override string VisitLeafVar(Var expr)
        {
            var name = GetNextName();
            _emitter.Input(name);
            Inputs.Add(name);
            return name;
        }

        protected override string VisitLeafFunction(Function expr) => ExprMemo[expr.Body];

        protected override string VisitLeafOp(Op expr) => string.Empty;

        protected override string VisitLeafTuple(IR.Tuple expr) => StringUtility.Join(",", expr.Fields.AsValueEnumerable().Select(x => ExprMemo[x]));

        protected override string VisitLeafCall(Call expr)
        {
            var name = GetNextName();
            switch (expr.Target)
            {
                case NcnnSoftmax op:
                    _emitter.Softmax(name, ExprMemo[expr.Arguments[0]], op.Axis);
                    break;
                default:
                    throw new NotSupportedException();
            }

            return name;
        }

        private string GetNextName() => $"layer{_layerId++}";
    }
}
