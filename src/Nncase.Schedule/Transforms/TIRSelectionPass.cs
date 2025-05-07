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
using Nncase.Utilities;

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
            var callers = func.Users.OfType<Call>().ToArray();
            var isEntry = callers.Length == 0;
            var visitor = new TIRSelectionVisitor(this, isEntry);
            (var newBody, var outBuffers) = visitor.Select(func);
            var inBuffers = func.Parameters.ToArray();

            if (isEntry)
            {
                // Allocate output buffers in the entry function
                var primFunc = new PrimFunction(
                    $"{input.Name}_prim",
                    ModuleKind,
                    newBody,
                    inBuffers.ToArray());
                return Task.FromResult((BaseFunction)primFunc);
            }
            else
            {
                // Allocate output buffers in the caller functions
                var primFunc = new PrimFunction(
                    $"{input.Name}_prim",
                    ModuleKind,
                    newBody,
                    inBuffers.Concat(outBuffers).ToArray());
                var primWrapper = new PrimFunctionWrapper(input.Name, primFunc, inBuffers.Length);
                AddOutputBufferAllocsToCallers(func, outBuffers, callers);
                return Task.FromResult((BaseFunction)primWrapper);
            }
        }

        return Task.FromResult(input);
    }

    protected abstract Expr SelectCall(Call call, IReadOnlyList<BaseExpr> arguments, Expr output);

    protected IRType GetArgumentType(BaseExpr argument)
    {
        return argument switch
        {
            TIR.Buffer b => b.DistributedType ?? b.CheckedType,
            _ => argument.CheckedType,
        };
    }

    private void AddOutputBufferAllocsToCallers(Function function, IEnumerable<Var> outputBuffers, IEnumerable<Call> callers)
    {
        var outputBufferShapes = outputBuffers.Select(x => (ElemType: x.CheckedDataType, Shape: x.CheckedShape)).ToArray();
        foreach (var caller in callers)
        {
            var outputAllocs = outputBufferShapes.Select(x => IR.F.Buffer.Uninitialized(x.ElemType, TIR.MemoryLocation.Data, x.Shape));
            var newArgs = caller.Arguments.ToArray().Concat(outputAllocs).ToArray();
            var newCaller = caller.With(arguments: newArgs);
            ReplaceUtility.ReplaceAllUsesWith(caller, newCaller);
        }
    }

    private sealed record SelectionResult(Sequential Body, IReadOnlyList<Var> OutputBuffers);

    private sealed class TIRSelectionVisitor : ExprCloner<Unit>
    {
        private readonly TIRSelectionPass _selectionPass;
        private readonly bool _isEntry;
        private readonly List<Expr> _body = new();
        private int _bufferIndex;

        public TIRSelectionVisitor(TIRSelectionPass selectionPass, bool isEntry)
        {
            _selectionPass = selectionPass;
            _isEntry = isEntry;
        }

        public SelectionResult Select(Function function)
        {
            Visit(function.Body, Unit.Default);

            var outBuffers = function.Body is IR.Tuple tuple
                ? tuple.Fields.AsValueEnumerable().Select(x => (Expr)ExprMemo[x]).ToArray()
                : [(Expr)ExprMemo[function.Body]];

            if (_isEntry)
            {
                _body.Add(T.Return(outBuffers));
                return new(new Sequential(_body.ToArray()), Array.Empty<Var>());
            }
            else
            {
                // Add necessary copy calls
                for (int i = 0; i < outBuffers.Length; i++)
                {
                    var previousBuffers = outBuffers.AsReadOnlySpan(0, i);
                    var currentBuffer = (Var)outBuffers[i];
                    if (previousBuffers.ReferenceContains(currentBuffer)
                        || function.Parameters.ReferenceContains(currentBuffer))
                    {
                        var newBuffer = currentBuffer.With($"out_{_bufferIndex++}");
                        _body.Add(T.Memcopy(newBuffer, currentBuffer));
                        outBuffers[i] = newBuffer;
                    }
                }

                return new(new Sequential(_body.ToArray()), outBuffers.Cast<Var>().ToArray());
            }
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

        private Expr SelectCall(Call call, IReadOnlyList<BaseExpr> arguments)
        {
            if (call.Target is IR.Tensors.GetItem && arguments[IR.Tensors.GetItem.Input.Index] is IR.Tuple tuple && call[IR.Tensors.GetItem.Index] is TensorConst index)
            {
                return (Expr)tuple[index.Value.ToScalar<int>()];
            }
            else
            {
                var output = CreateOutputBuffer(call);
                var newCall = call.Target switch
                {
                    PrimFunctionWrapper { Target: TIR.PrimFunction deviceFunc } => new Call(deviceFunc, arguments.Append(output).ToArray()),
                    Function fn => new Call(new FunctionWrapper(_selectionPass.ModuleKind, fn), arguments.Append(output).ToArray()),
                    _ => _selectionPass.SelectCall(call, arguments, output),
                };
                _body.Add(newCall);
                return output;
            }
        }

        private Expr CreateOutputBuffer(Call expr)
        {
            var root = VisitRoot!;
            var memoryLocation = MemoryLocation.Data;
            string namePrefix = "call_";
            if (ReferenceEquals(root, expr)
                || (root is IR.Tuple tuple && tuple.Fields.AsValueEnumerable().Contains(expr, ReferenceEqualityComparer.Instance)))
            {
                namePrefix = "out_";
                if (_isEntry)
                {
                    memoryLocation = MemoryLocation.Output;
                }
                else
                {
                    return new Var($"{namePrefix}{_bufferIndex++}", expr.CheckedType);
                }
            }

            return T.CreateBuffer(expr.CheckedTensorType, memoryLocation, out _, $"{namePrefix}{_bufferIndex++}", expr.CheckedType as DistributedType);
        }
    }
}
