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
            var callers = func.Users.Where(x => x is Call or FunctionWrapper).ToArray();
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
                AddOutputBufferAllocsToCallers(func, outBuffers, callers.OfType<Call>());
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

            var outBuffers = function.Body switch
            {
                IR.Tuple tuple => tuple.Fields.AsValueEnumerable().Select(x => (Expr)ExprMemo[x]).ToArray(),
                var body => ExprMemo[function.Body] switch
                {
                    IR.Tuple bodyTuple => bodyTuple.Fields.AsValueEnumerable().Select(x => (Expr)x).ToArray(),
                    var x => [(Expr)x],
                },
            };

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
                    var currentBuffer = outBuffers[i];
                    if (currentBuffer is not Var
                        || previousBuffers.ReferenceContains(currentBuffer)
                        || function.Parameters.ReferenceContains((Var)currentBuffer))
                    {
                        var newBuffer = new Var($"out_{_bufferIndex++}", _selectionPass.GetArgumentType(currentBuffer));
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

        protected sealed override BaseExpr VisitLeafCall(Call expr, Unit context)
        {
            var args = expr.Arguments.AsValueEnumerable().Select(x => ExprMemo[x]).ToArray();
            return SelectCall(expr, args);
        }

        protected override BaseExpr VisitLeafIf(If expr, Unit context)
        {
            var output = CreateOutputBuffer(expr);
            var condition = (Expr)Visit(expr.Condition, context);
            return T.Let(out var outputVar, (Expr)output).Body(
                T.Assign(out var arguments, expr.Arguments.AsValueEnumerable().Select(x => ExprMemo[x]).ToArray().Append(outputVar).ToArray()),
                T.If(condition)
                    .Then(new Call(new FunctionWrapper(_selectionPass.ModuleKind, expr.Then), arguments))
                    .Else(new Call(new FunctionWrapper(_selectionPass.ModuleKind, expr.Else), arguments)))
                .Build();
        }

        private BaseExpr SelectCall(Call call, IReadOnlyList<BaseExpr> arguments)
        {
            if (call.Target is IR.Tensors.GetItem && arguments[IR.Tensors.GetItem.Input.Index] is IR.Tuple tuple && call[IR.Tensors.GetItem.Index] is DimConst index)
            {
                return tuple[index.Value];
            }
            else
            {
                var output = CreateOutputBuffer(call);
                var newCall = call.Target switch
                {
                    PrimFunctionWrapper { Target: TIR.PrimFunction deviceFunc } => new Call(deviceFunc, arguments.Append(output).ToArray()),
                    Function fn => new Call(new FunctionWrapper(_selectionPass.ModuleKind, fn), arguments.Append(output).ToArray()),
                    _ => _selectionPass.SelectCall(call, arguments, (Expr)output),
                };
                _body.Add(newCall);
                return output;
            }
        }

        private BaseExpr CreateOutputBuffer(Expr expr)
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

            if (expr.CheckedType is TupleType tt)
            {
                var fields = tt.Fields.AsValueEnumerable().Select(x => CreateBuffer(x, memoryLocation)).ToArray();
                return new IR.Tuple(fields);
            }
            else
            {
                return CreateBuffer(expr.CheckedType, memoryLocation);
            }
        }

        private TIR.Buffer CreateBuffer(IRType type, MemoryLocation memoryLocation)
        {
            var tensorType = type switch
            {
                DistributedType dt => dt.TensorType,
                TensorType tt => tt,
                _ => throw new ArgumentException($"Unsupported type: {type}"),
            };
            return T.CreateBuffer(tensorType, memoryLocation, out _, $"buffer_{_bufferIndex++}", type as DistributedType);
        }
    }
}
