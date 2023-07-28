// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Reactive;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.TIR;

namespace Nncase.Evaluator;

internal sealed class TypeInferenceVisitor : ExprVisitor<IRType, Unit>
{
    private readonly TypeInferenceContext _context;
    private readonly Dictionary<Type, ITypeInferencer> _inferencer_cache;

    public TypeInferenceVisitor()
    {
        _context = new TypeInferenceContext();
        _inferencer_cache = new Dictionary<Type, ITypeInferencer>();
    }

    /// <summary>
    /// Gets a value indicating whether is fully inferenced.
    /// </summary>
    public bool IsFullyInferenced { get; private set; } = true;

    public override void Clear()
    {
        IsFullyInferenced = true;
        base.Clear();
    }

    /// <inheritdoc/>
    protected override IRType VisitLeafBlock(Block expr)
    {
        for (int i = 0; i < expr.IterVars.Length; i++)
        {
            VerifySubField(expr, expr.IterVars[i], TypePatternUtility.IsIntegralScalar());
        }

        VerifySubField(expr, expr.InitBody, TypePatternUtility.IsUnit());
        VerifySubField(expr, expr.Body, TypePatternUtility.IsUnit());
        VerifySubField(expr, expr.Predicate, TypePatternUtility.IsBoolScalar());

        return TupleType.Void;
    }

    /// <inheritdoc/>
    protected override IRType VisitLeafBufferLoad(BufferLoad expr)
    {
        IRType type;
        VerifySubField(expr, expr.Buffer, TypePatternUtility.IsPointer());
        for (int i = 0; i < expr.Indices.Length; i++)
        {
            VerifySubField(expr, expr.Indices[i], TypePatternUtility.IsIntegralScalar(), $"BufferLoad.Indices[{i}]");
        }

        if (expr.Buffer.CheckedType is TensorType { IsScalar: true, DType: PointerType { ElemType: PrimType pointedType } })
        {
            type = TensorType.Scalar(pointedType);
        }
        else
        {
            type = new InvalidType($"Can't load from {expr.Buffer.CheckedType}");
        }

        return type;
    }

    /// <inheritdoc/>
    protected override IRType VisitLeafBufferRegion(BufferRegion expr)
    {
        VerifySubField(expr, expr.Buffer, TypePatternUtility.IsTensor());
        foreach (var r in expr.Region)
        {
            VerifySubField(expr, r.Start, TypePatternUtility.IsIntegralScalar());
            VerifySubField(expr, r.Stop, TypePatternUtility.IsIntegralScalar());
            VerifySubField(expr, r.Stop, TypePatternUtility.IsIntegralScalar());
        }

        // TODO: need infer the sub region shape/stride
        var type = expr.Buffer.CheckedType;
        return type;
    }

    /// <inheritdoc/>
    protected override IRType VisitLeafBufferStore(BufferStore expr)
    {
        VerifySubField(expr, expr.Buffer, TypePatternUtility.IsPointer());
        for (int i = 0; i < expr.Indices.Length; i++)
        {
            VerifySubField(expr, expr.Indices[i], TypePatternUtility.IsIntegralScalar(), $"BufferStore.Indices[{i}]");
        }

        VerifySubField(expr, expr.Value, TypePatternUtility.IsScalar());

        IRType type;
        if (expr.Value.CheckedType is TensorType { IsScalar: true, DType: PrimType valueType } &&
            expr.Buffer.CheckedType is TensorType { IsScalar: true, DType: PointerType { ElemType: PrimType pointedType } }
            && valueType == pointedType)
        {
            type = TupleType.Void;
        }
        else
        {
            type = new InvalidType($"Can't store {expr.Value.CheckedType} to {expr.Buffer.CheckedType}");
        }

        return type;
    }

    /// <inheritdoc/>
    protected override IRType VisitLeafCall(Call expr)
    {
        _context.CurrentCall = expr;
        var type = expr.Target switch
        {
            Op op => CompilerServices.InferenceOp(op, _context, _inferencer_cache),
            BaseFunction func => BaseFunctionInfer(expr, func),
            _ => new InvalidType("Target of call expression should be either a function or an op."),
        };
        _context.CurrentCall = null;
        return type;
    }

    private IRType BaseFunctionInfer(Call call, BaseFunction func)
    {
        if (func.CheckedType is InvalidType)
        {
            return func.CheckedType;
        }

        return ((CallableType)func.CheckedType).ReturnType;
    }

    /// <inheritdoc/>
    protected override IRType VisitLeafConst(Const expr)
    {
        var type = expr.ValueType;
        return type;
    }

    /// <inheritdoc/>
    protected override IRType VisitLeafFor(For expr)
    {
        VerifySubField(expr, expr.Domain.Start, TypePatternUtility.IsIntegralScalar());
        VerifySubField(expr, expr.Domain.Stop, TypePatternUtility.IsIntegralScalar());
        VerifySubField(expr, expr.LoopVar, TypePatternUtility.IsIntegralScalar());
        VerifySubField(expr, expr.Body, TypePatternUtility.IsUnit());

        var type = TupleType.Void;
        return type;
    }

    /// <inheritdoc/>
    protected override IRType VisitLeafFunction(Function expr)
    {
        foreach (var p in expr.Parameters)
        {
            VerifySubField(expr, p);
        }

        VerifySubField(expr, expr.Body);

        var paramTypes = expr.Parameters.AsValueEnumerable().Select(x => x.CheckedType).ToArray();
        var type = new CallableType(expr.Body.CheckedType, ImmutableArray.Create(paramTypes));
        return type;
    }

    /// <inheritdoc/>
    protected override IRType VisitLeafFusion(Fusion expr)
    {
        foreach (var p in expr.Parameters)
        {
            VerifySubField(expr, p);
        }

        VerifySubField(expr, expr.Body);

        var paramTypes = expr.Parameters.AsValueEnumerable().Select(x => x.CheckedType).ToArray();
        var type = new CallableType(expr.Body.CheckedType, ImmutableArray.Create(paramTypes));
        return type;
    }

    /// <inheritdoc/>
    protected override IRType VisitLeafIf(If expr)
    {
        return TypeInference.CommonType(expr.Then.CheckedType, expr.Else.CheckedType);
    }

    /// <inheritdoc/>
    protected override IRType VisitLeafIfThenElse(IfThenElse expr)
    {
        VerifySubField(expr, expr.Condition, TypePatternUtility.IsBoolScalar());
        VerifySubField(expr, expr.Then, TypePatternUtility.IsUnit());
        VerifySubField(expr, expr.Else, TypePatternUtility.IsUnit());

        IRType type = TupleType.Void;
        return type;
    }

    /// <inheritdoc/>
    protected override IRType VisitLeafIterVar(IterVar expr)
    {
        VerifySubField(expr, expr.Value, TypePatternUtility.IsScalar() & TypePatternUtility.HasDataType(DataTypes.Int32));
        VerifySubField(expr, expr.Dom.Start, TypePatternUtility.IsScalar() & TypePatternUtility.HasDataType(DataTypes.Int32));
        VerifySubField(expr, expr.Dom.Stop, TypePatternUtility.IsScalar() & TypePatternUtility.HasDataType(DataTypes.Int32));
        VerifySubField(expr, expr.Dom.Step, TypePatternUtility.IsScalar() & TypePatternUtility.HasDataType(DataTypes.Int32));

        var type = TensorType.Scalar(DataTypes.Int32);
        return type;
    }

    /// <inheritdoc/>
    protected override IRType VisitLeafLet(Let expr)
    {
        if (expr.Var.CheckedType != expr.Expression.CheckedType)
        {
            throw new TypeInferenceInterruptException(new InvalidType("Var Type != Expression Type"));
        }

        VerifySubField(expr, expr.Body, TypePatternUtility.IsUnit());

        IRType type = TupleType.Void;
        return type;
    }

    /// <inheritdoc/>
    protected override IRType VisitLeafLogicalBuffer(LogicalBuffer expr)
    {
        var type = new TensorType(expr.ElemType, Shape.Unknown(expr.Rank));
        return type;
    }

    /// <inheritdoc/>
    protected override IRType VisitLeafMarker(Marker expr)
    {
        var type = expr.Target.CheckedType;
        return type;
    }

    /// <inheritdoc/>
    protected override IRType VisitLeafNone(None expr)
    {
        var type = NoneType.Default;
        return type;
    }

    /// <inheritdoc/>
    protected override IRType VisitLeafOp(Op expr)
    {
        var paramTypes = expr.Parameters.Select(_ => (IRType)AnyType.Default).ToArray();
        var type = new CallableType(AnyType.Default, ImmutableArray.Create(paramTypes));
        return type;
    }

    /// <inheritdoc/>
    protected override IRType VisitLeafPhysicalBuffer(PhysicalBuffer expr)
    {
        var type = new TensorType(expr.ElemType, new(expr.FixedDimensions));
        return type;
    }

    /// <inheritdoc/>
    protected override IRType VisitLeafPrimFunction(PrimFunction expr)
    {
        foreach (var p in expr.Parameters)
        {
            VerifySubField(expr, p);
        }

        VerifySubField(expr, expr.Body);

        var paramTypes = expr.Parameters.AsValueEnumerable().Select(x => x.CheckedType).ToArray();
        var type = new CallableType(expr.Body.CheckedType, ImmutableArray.Create(paramTypes));
        return type;
    }

    /// <inheritdoc/>
    protected override IRType VisitLeafPrimFunctionWrapper(PrimFunctionWrapper expr)
    {
        var type = new CallableType(expr.ReturnType, new(expr.ParameterTypes));
        return type;
    }

    /// <inheritdoc/>
    protected override IRType VisitLeafRange(Nncase.TIR.Range expr)
    {
        var type = new TupleType(ImmutableArray.Create(expr.Start.CheckedType, expr.Stop.CheckedType, expr.Step.CheckedType));
        return type;
    }

    /// <inheritdoc/>
    protected override IRType VisitLeafSequential(Sequential expr)
    {
        for (int i = 0; i < expr.Fields.Length; i++)
        {
            VerifySubField(expr, expr.Fields[i], null, $"Sequential Line {i}");
        }

        var type = TupleType.Void;
        return type;
    }

    /// <inheritdoc/>
    protected override IRType VisitLeafTuple(IR.Tuple expr)
    {
        for (int i = 0; i < expr.Fields.Length; i++)
        {
            VerifySubField(expr, expr.Fields[i], null, $"IR.Tuple Item {i}");
        }

        var type = new TupleType(expr.Fields.AsValueEnumerable().Select(f => f.CheckedType).ToArray());
        return type;
    }

    /// <inheritdoc/>
    protected override IRType VisitLeafVar(Var expr)
    {
        var type = expr.TypeAnnotation;
        return type;
    }

    /// <inheritdoc/>
    protected override IRType VisitLet(Let expr)
    {
        if (HasVisited(expr, out var type))
        {
            return type;
        }

        Visit(expr.Expression);

        if (expr.Var.TypeAnnotation is not AnyType)
        {
            // now we need custom visit the var.
            type = new InvalidType("The Let Bind Var Must Be Any Type!");
            SetCheckedType(expr.Var, type);
            SetCheckedType(expr.Body, type);
            return type;
        }
        else
        {
            // now change the var checkedtype
            SetCheckedType(expr.Var, expr.Expression.CheckedType);
            Visit(expr.Body);
            return VisitLeafLet(expr);
        }
    }

    protected override IRType DispatchVisit(Expr expr)
    {
        if (IRHelpers.GetRawCheckedType(expr) is null)
        {
            try
            {
                SetCheckedType(expr, base.DispatchVisit(expr));
            }
            catch (TypeInferenceInterruptException e)
            {
                SetCheckedType(expr, e.ReasonType);
            }
        }

        return expr.CheckedType;
    }

    /// <summary>
    /// Verify the expression sub field type is valid.
    /// </summary>
    private void VerifySubField(Expr parent, Expr field, TypePattern? pattern = null, [CallerArgumentExpression("field")] string? exprMsg = null)
    {
        pattern ??= TypePatternUtility.IsIRType();
        if (field.CheckedType is InvalidType invalidType)
        {
            throw new TypeInferenceInterruptException(new InvalidType($"Invalid {exprMsg} <== {invalidType.Reason}"));
        }
        else if (field.CheckedType is AnyType)
        {
            return;
        }
        else if (!pattern.MatchLeaf(field.CheckedType!))
        {
            throw new TypeInferenceInterruptException(new InvalidType($"The {exprMsg} Require {pattern.Reason}"));
        }
    }

    /// <summary>
    /// set expr's current type.
    /// </summary>
    private void SetCheckedType(Expr expr, IRType type)
    {
        // note can't determine whether to update checkedtype
        // eg. old call[x,y] shape is [5,6]
        //     new call[x,y] shape is [5,6,1,1] we can't compare the two ir type.
        IRHelpers.SetRawCheckedType(expr, type);
        IsFullyInferenced &= type is not InvalidType;
    }
}
