// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.TIR;

namespace Nncase.Passes.Transforms;

public abstract class TIRSelectionPass : FunctionPass
{
    public TIRSelectionPass(string moduleKind)
    {
        ModuleKind = moduleKind;
    }

    public string ModuleKind { get; }

    protected override Task<BaseFunction> RunCoreAsync(BaseFunction input, RunPassContext context)
    {
        if (input.ModuleKind == ModuleKind
            && input is Function func)
        {
            var visitor = new TIRSelectionVisitor(this);
            (var newBody, var outBuffers) = visitor.Select(func);

            var inBuffers = func.Parameters.ToArray();
            var outputBufferShapes = outBuffers.Select(x => (ElemType: x.CheckedDataType, Shape: x.CheckedShape.ToValueArrayExpr())).ToArray();
            var primFunc = new PrimFunction(
                $"{input.Name}_prim",
                ModuleKind,
                newBody,
                inBuffers.Concat(outBuffers).ToArray());
            var primWrapper = new PrimFunctionWrapper(input.Name, primFunc, inBuffers.Length);
            AddOutputBufferAllocsToCallers(func, outBuffers);
            return Task.FromResult((BaseFunction)primWrapper);
        }

        return Task.FromResult(input);
    }

    protected abstract Expr SelectCall(Call call, IReadOnlyList<Expr> arguments, Expr output);

    protected IRType GetArgumentType(Expr argument)
    {
        return argument switch
        {
            TIR.Buffer b => b.DistributedType ?? b.CheckedType,
            _ => argument.CheckedType,
        };
    }

    private void AddOutputBufferAllocsToCallers(Function function, IEnumerable<Var> outputBuffers)
    {
        var callers = function.Users.OfType<Call>().ToArray();
        var outputBufferShapes = outputBuffers.Select(x => (ElemType: x.CheckedDataType, Shape: x.CheckedShape.ToValueArrayExpr())).ToArray();
        foreach (var caller in callers)
        {
            var outputAllocs = outputBufferShapes.Select(x => IR.F.Buffer.Uninitialized(x.ElemType, TIR.MemoryLocation.Data, x.Shape));
            var newArgs = caller.Arguments.ToArray().Concat(outputAllocs).ToArray();
            var newCaller = caller.With(arguments: newArgs);
            IRHelpers.ReplaceAllUsesWith(caller, newCaller);
        }
    }

    private sealed record SelectionResult(Sequential Body, IReadOnlyList<Var> OutputBuffers);

    private sealed class TIRSelectionVisitor : ExprCloner<Unit>
    {
        private readonly TIRSelectionPass _selectionPass;
        private readonly List<Expr> _body = new();
        private int _bufferIndex;

        public TIRSelectionVisitor(TIRSelectionPass selectionPass)
        {
            _selectionPass = selectionPass;
        }

        public SelectionResult Select(Function function)
        {
            Visit(function.Body, Unit.Default);

            // Add necessary copy calls
            var outBuffers = function.Body is IR.Tuple tuple
                ? tuple.Fields.AsValueEnumerable().Select(x => (Var)ExprMemo[x]).ToArray()
                : [(Var)ExprMemo[function.Body]];

            for (int i = 0; i < outBuffers.Length; i++)
            {
                var previousBuffers = outBuffers.AsSpan(0, i);
                var currentBuffer = outBuffers[i];
                if (previousBuffers.Contains(currentBuffer)
                    || function.Parameters.Contains(currentBuffer))
                {
                    var newBuffer = currentBuffer.With($"out_{_bufferIndex++}");
                    _body.Add(T.Memcopy(newBuffer, currentBuffer));
                    outBuffers[i] = newBuffer;
                }
            }

            return new(new Sequential(_body.ToArray()), outBuffers);
        }

        protected sealed override Expr VisitLeafTensorConst(TensorConst expr, Unit context)
        {
            return T.AttachBuffer(expr, out _, $"const_{_bufferIndex++}");
        }

        protected sealed override Expr VisitLeafVar(Var expr, Unit context) => expr;

        protected sealed override Expr VisitLeafCall(Call expr, Unit context)
        {
            var args = expr.Arguments.AsValueEnumerable().Select(x => ExprMemo[x]).ToArray();
            return SelectCall(expr, args);
        }

        private Expr SelectCall(Call call, IReadOnlyList<Expr> arguments)
        {
            if (call.Target is IR.Tensors.GetItem && arguments[IR.Tensors.GetItem.Input.Index] is IR.Tuple tuple && call[IR.Tensors.GetItem.Index] is TensorConst index)
            {
                return tuple[index.Value.ToScalar<int>()];
            }
            else
            {
                var output = CreateOutputBuffer(call);
                var newCall = call.Target switch
                {
                    PrimFunctionWrapper { Target: TIR.PrimFunction deviceFunc } => new Call(deviceFunc, arguments.Append(output).ToArray()),
                    _ => _selectionPass.SelectCall(call, arguments, output),
                };
                _body.Add(newCall);
                return output;
            }
        }

        private Expr CreateOutputBuffer(Call expr)
        {
            var root = VisitRoot!;
            if (ReferenceEquals(root, expr)
                || (root is IR.Tuple tuple && tuple.Fields.AsValueEnumerable().Contains(expr, ReferenceEqualityComparer.Instance)))
            {
                return new Var($"out_{_bufferIndex++}", expr.CheckedType);
            }
            else
            {
                return T.CreateBuffer(expr.CheckedTensorType, MemoryLocation.Data, out _, $"call_{_bufferIndex++}", expr.CheckedType as DistributedType);
            }
        }
    }
}
