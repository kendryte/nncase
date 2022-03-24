// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.TIR;

namespace Nncase.Evaluator;

internal sealed class TypeInferenceVisitor : ExprVisitor<IRType, IRType>
{
    private readonly TypeInferenceContext _context;

    public TypeInferenceVisitor()
    {
        _context = new TypeInferenceContext(ExpressionMemo);
    }

    /// <summary>
    /// Gets a value indicating whether is fully inferenced.
    /// </summary>
    public bool IsFullyInferenced { get; private set; } = true;

    /// <inheritdoc/>
    public override IRType VisitLeaf(Call expr)
    {
        _context.CurrentCall = expr;
        var type = expr.Target switch
        {
            Function func => ((CallableType)Visit(func)).ReturnType,
            Op op => CompilerServices.InferenceOp(op, _context),
            PrimFunction primfunc => ((CallableType)Visit(primfunc)).ReturnType,
            _ => new InvalidType("Target of call expression should be either a function or an op."),
        };
        _context.CurrentCall = null;
        SetCheckedType(expr, type);
        return type;
    }

    /// <inheritdoc/>
    public override IRType VisitLeaf(Const expr)
    {
        var type = expr.ValueType;
        SetCheckedType(expr, type);
        return type;
    }

    /// <inheritdoc/>
    public override IRType VisitLeaf(Function expr)
    {
        foreach (var p in expr.Parameters) { VerifySubField(expr, p); }
        VerifySubField(expr, expr.Body);
        if (expr.CheckedType is not null) { return expr.CheckedType; }

        var paramTypes = expr.Parameters.Select(Visit).ToArray();
        var type = new CallableType(expr.Body is Sequential seq ? (seq.Count == 0 ? TupleType.Void : Visit(seq.Last())) : Visit(expr.Body), ImmutableArray.Create(paramTypes));
        SetCheckedType(expr, type);
        return type;
    }

    /// <inheritdoc/>
    public override IRType VisitLeaf(PrimFunction expr)
    {
        foreach (var p in expr.Parameters) { VerifySubField(expr, p); }
        VerifySubField(expr, expr.Body);
        if (expr.CheckedType is not null) { return expr.CheckedType; }

        var paramTypes = expr.Parameters.Select(Visit).ToArray();
        var type = new CallableType(expr.Body is Sequential seq ? (seq.Count == 0 ? TupleType.Void : Visit(seq.Last())) : Visit(expr.Body), ImmutableArray.Create(paramTypes));
        SetCheckedType(expr, type);
        return type;
    }

    /// <inheritdoc/>
    public override IRType VisitLeaf(Op expr)
    {
        var paramTypes = expr.Parameters.Select(_ => (IRType)AnyType.Default).ToArray();
        var type = new CallableType(AnyType.Default, ImmutableArray.Create(paramTypes));
        SetCheckedType(expr, type);
        return type;
    }

    /// <inheritdoc/>
    public override IRType VisitLeaf(IR.Tuple expr)
    {
        var fieldTypes = expr.Fields.Select(Visit).ToArray();
        var type = new TupleType(ImmutableArray.Create(fieldTypes));
        SetCheckedType(expr, type);
        return type;
    }

    /// <inheritdoc/>
    public override IRType VisitLeaf(Var expr)
    {
        var type = expr.TypeAnnotation ?? AnyType.Default;
        SetCheckedType(expr, type);
        return type;
    }

    /// <summary>
    /// Verify the expression sub field type is valid.
    /// </summary>
    /// <param name="parent"></param>
    /// <param name="expr"></param>
    /// <param name="exprMsg"></param>
    void VerifySubField(Expr parent, Expr expr, TypePattern? pattern = null, [CallerArgumentExpression("expr")] string? exprMsg = null)
    {
        pattern ??= TypePatternUtility.IsIRType();
        if (parent.CheckedType is null)
        {
            if (expr.CheckedType is InvalidType invalidType)
            {
                SetCheckedType(parent, new InvalidType($"Invalid {exprMsg} <== {invalidType.Reason}"));
                return;
            }
            else if (expr.CheckedType is AnyType any)
            {
                SetCheckedType(parent, any);
            }
            else if (!pattern.MatchLeaf(expr.CheckedType!))
            {
                SetCheckedType(parent, new InvalidType($"The {exprMsg} Require {pattern.Reason}"));
            }
        }
    }

    /// <inheritdoc/>
    public override IRType VisitLeaf(IterVar expr)
    {
        VerifySubField(expr, expr.Value);
        VerifySubField(expr, expr.Dom.Start);
        VerifySubField(expr, expr.Dom.Stop);
        if (expr.CheckedType is not null) { return expr.CheckedType; }
        var type = expr.TypeAnnotation;
        SetCheckedType(expr, type);
        return type;
    }

    /// <inheritdoc/>
    public override IRType VisitLeaf(Sequential expr)
    {
        IRType type;
        foreach (var i in Enumerable.Range(0, expr.Fields.Count))
        {
            VerifySubField(expr, expr.Fields[i], null, $"Sequential Line {i}");
        }

        if (expr.CheckedType is not null) { return expr.CheckedType; }
        type = TupleType.Void;
        SetCheckedType(expr, type);
        return type;
    }

    /// <inheritdoc/>
    public override IRType VisitLeaf(For expr)
    {
        IRType type;
        VerifySubField(expr, expr.Dom.Start, TypePatternUtility.IsIntegralScalar());
        VerifySubField(expr, expr.Dom.Stop, TypePatternUtility.IsIntegralScalar());
        VerifySubField(expr, expr.LoopVar, TypePatternUtility.IsIntegralScalar());
        VerifySubField(expr, expr.Body, TypePatternUtility.IsUnit());
        if (expr.CheckedType is not null) { return expr.CheckedType; }
        type = TupleType.Void;
        SetCheckedType(expr, type);
        return type;
    }

    /// <inheritdoc/>
    public override IRType VisitLeaf(Block expr)
    {
        IRType type;
        foreach (var i in Enumerable.Range(0, expr.IterVars.Count))
        {
            VerifySubField(expr, expr.IterVars[i], TypePatternUtility.IsIntegralScalar());
        }

        VerifySubField(expr, expr.InitBody, TypePatternUtility.IsUnit());
        VerifySubField(expr, expr.Body, TypePatternUtility.IsUnit());
        VerifySubField(expr, expr.Predicate, TypePatternUtility.IsBoolScalar());
        if (expr.CheckedType is not null) { return expr.CheckedType; }
        type = TupleType.Void;
        SetCheckedType(expr, type);
        return type;
    }

    public override IRType VisitLeaf(BufferLoad expr)
    {
        VerifySubField(expr, expr.Buffer, TypePatternUtility.IsPointer());
        foreach (var i in Enumerable.Range(0, expr.Indices.Count)) { VerifySubField(expr, expr.Indices[i], TypePatternUtility.IsIntegralScalar()); }
        if (expr.CheckedType is not null) { return expr.CheckedType; }
        IRType type;
        if (expr.Buffer.CheckedType is TensorType { IsScalar: true, DType: PointerType { ElemType: PrimType pointedType } })
        {
            type = TensorType.Scalar(pointedType);
        }
        else
        {
            type = new InvalidType($"Can't Load From {expr.Buffer.CheckedType}");
        }
        SetCheckedType(expr, type);
        return type;
    }

    public override IRType VisitLeaf(BufferStore expr)
    {
        VerifySubField(expr, expr.Buffer, TypePatternUtility.IsPointer());
        foreach (var i in Enumerable.Range(0, expr.Indices.Count)) { VerifySubField(expr, expr.Indices[i], TypePatternUtility.IsIntegralScalar()); }
        VerifySubField(expr, expr.Value, TypePatternUtility.IsScalar());

        if (expr.CheckedType is not null) { return expr.CheckedType; }
        IRType type;
        if (expr.Value.CheckedType is TensorType { IsScalar: true, DType: PrimType valueType } &&
            expr.Buffer.CheckedType is TensorType { IsScalar: true, DType: PointerType { ElemType: PrimType pointedType } }
            && valueType == pointedType)
        {
            type = TupleType.Void;
        }
        else
        {
            type = new InvalidType($"Can't Store {expr.Value.CheckedType} To {expr.Buffer.CheckedType}");
        }

        SetCheckedType(expr, type);
        return type;
    }

    /// <inheritdoc/>
    public override IRType VisitLeaf(IfThenElse expr)
    {
        VerifySubField(expr, expr.Condition, TypePatternUtility.IsBoolScalar());
        VerifySubField(expr, expr.Then, TypePatternUtility.IsUnit());
        VerifySubField(expr, expr.Else, TypePatternUtility.IsUnit());
        if (expr.CheckedType is not null) { return expr.CheckedType; }
        IRType type = TupleType.Void;
        SetCheckedType(expr, type);
        return type;
    }

    /// <inheritdoc/>
    public override IRType VisitLeaf(Let expr)
    {
        VerifySubField(expr, expr.Var, TypePatternUtility.IsPointer());
        VerifySubField(expr, expr.Expression, TypePatternUtility.IsPointer());
        VerifySubField(expr, expr.Body, TypePatternUtility.IsUnit());
        if (expr.CheckedType is not null) { return expr.CheckedType; }
        IRType type = TupleType.Void;
        SetCheckedType(expr, type);
        return type;
    }

    /// <inheritdoc/>
    public override IRType VisitLeaf(Nncase.TIR.Buffer expr)
    {
        if (expr.CheckedType is not null) { return expr.CheckedType; }
        IRType type = TensorType.Pointer(expr.ElemType.DType);
        SetCheckedType(expr, type);
        return type;
    }

    /// <summary>
    /// set expr's current type.
    /// </summary>
    /// <param name="expr"></param>
    /// <param name="type"></param>
    private void SetCheckedType(Expr expr, IRType type)
    {
        expr.CheckedType = type;
        IsFullyInferenced &= type is not (AnyType or InvalidType);
    }
}
