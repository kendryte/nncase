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
            PrimFunctionWrapper func => ((CallableType)Visit(func)).ReturnType,
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
        try
        {
            foreach (var p in expr.Parameters) { VerifySubField(expr, p); }
            VerifySubField(expr, expr.Body);
        }
        catch (TypeInferenceInterruptException e)
        {
            SetCheckedType(expr, e.ReasonType);
            return e.ReasonType;
        }

        var paramTypes = expr.Parameters.Select(Visit).ToArray();
        var type = new CallableType(Visit(expr.Body), ImmutableArray.Create(paramTypes));
        SetCheckedType(expr, type);
        return type;
    }

    /// <inheritdoc/>
    public override IRType VisitLeaf(PrimFunctionWrapper expr)
    {
        var type = new CallableType(expr.ReturnType!, new(expr.ParameterTypes!));
        SetCheckedType(expr, type);
        return type;
    }

    /// <inheritdoc/>
    public override IRType VisitLeaf(PrimFunction expr)
    {
        try
        {
            foreach (var p in expr.Parameters) { VerifySubField(expr, p); }
            VerifySubField(expr, expr.Body);
        }
        catch (TypeInferenceInterruptException e)
        {
            SetCheckedType(expr, e.ReasonType);
            return e.ReasonType;
        }

        var paramTypes = expr.Parameters.Select(Visit).ToArray();
        var type = new CallableType(Visit(expr.Body), ImmutableArray.Create(paramTypes));
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
        try
        {
            foreach (var i in Enumerable.Range(0, expr.Fields.Count))
            {
                VerifySubField(expr, expr.Fields[i], null, $"IR.Tuple Item {i}");
            }
        }
        catch (TypeInferenceInterruptException e)
        {
            SetCheckedType(expr, e.ReasonType);
            return e.ReasonType;
        }

        var type = new TupleType(expr.Fields.Select(f => f.CheckedType!).ToArray());
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

    /// <inheritdoc/>
    public override IRType VisitLeaf(None expr)
    {
        var type = NoneType.Default;
        SetCheckedType(expr, type);
        return type;
    }

    /// <inheritdoc/>
    public override IRType VisitLeaf(Marker expr)
    {
        var type = Visit(expr.Target);
        SetCheckedType(expr, type);
        return type;
    }

    /// <summary>
    /// Verify the expression sub field type is valid.
    /// </summary>
    /// <param name="parent"></param>
    /// <param name="field"></param>
    /// <param name="exprMsg"></param>
    void VerifySubField(Expr parent, Expr field, TypePattern? pattern = null, [CallerArgumentExpression("expr")] string? exprMsg = null)
    {
        pattern ??= TypePatternUtility.IsIRType();

        if (field.CheckedType is InvalidType invalidType)
        {
            throw new TypeInferenceInterruptException(new InvalidType($"Invalid {exprMsg} <== {invalidType.Reason}"));
        }
        else if (field.CheckedType is AnyType any)
        {
            return;
        }
        else if (!pattern.MatchLeaf(field.CheckedType!))
        {
            throw new TypeInferenceInterruptException(new InvalidType($"The {exprMsg} Require {pattern.Reason}"));
        }
    }

    /// <inheritdoc/>
    public override IRType VisitLeaf(IterVar expr)
    {
        try
        {
            VerifySubField(expr, expr.Value, TypePatternUtility.IsScalar() & TypePatternUtility.HasDataType(DataTypes.Int32));
            VerifySubField(expr, expr.Dom.Start, TypePatternUtility.IsScalar() & TypePatternUtility.HasDataType(DataTypes.Int32));
            VerifySubField(expr, expr.Dom.Stop, TypePatternUtility.IsScalar() & TypePatternUtility.HasDataType(DataTypes.Int32));
            VerifySubField(expr, expr.Dom.Step, TypePatternUtility.IsScalar() & TypePatternUtility.HasDataType(DataTypes.Int32));
        }
        catch (TypeInferenceInterruptException e)
        {
            SetCheckedType(expr, e.ReasonType);
            return e.ReasonType;
        }
        var type = TensorType.Scalar(DataTypes.Int32);
        SetCheckedType(expr, type);
        return type;
    }

    /// <inheritdoc/>
    public override IRType VisitLeaf(Sequential expr)
    {
        try
        {
            foreach (var i in Enumerable.Range(0, expr.Fields.Count))
            {
                VerifySubField(expr, expr.Fields[i], null, $"Sequential Line {i}");
            }
        }
        catch (TypeInferenceInterruptException e)
        {
            SetCheckedType(expr, e.ReasonType);
            return e.ReasonType;
        }
        var type = TupleType.Void;
        SetCheckedType(expr, type);
        return type;
    }

    /// <inheritdoc/>
    public override IRType VisitLeaf(For expr)
    {
        try
        {
            VerifySubField(expr, expr.Domain.Start, TypePatternUtility.IsIntegralScalar());
            VerifySubField(expr, expr.Domain.Stop, TypePatternUtility.IsIntegralScalar());
            VerifySubField(expr, expr.LoopVar, TypePatternUtility.IsIntegralScalar());
            VerifySubField(expr, expr.Body, TypePatternUtility.IsUnit());
        }
        catch (TypeInferenceInterruptException e)
        {
            SetCheckedType(expr, e.ReasonType);
            return e.ReasonType;
        }
        var type = TupleType.Void;
        SetCheckedType(expr, type);
        return type;
    }

    /// <inheritdoc/>
    public override IRType VisitLeaf(Block expr)
    {
        try
        {
            foreach (var i in Enumerable.Range(0, expr.IterVars.Count))
            {
                VerifySubField(expr, expr.IterVars[i], TypePatternUtility.IsIntegralScalar());
            }
            VerifySubField(expr, expr.InitBody, TypePatternUtility.IsUnit());
            VerifySubField(expr, expr.Body, TypePatternUtility.IsUnit());
            VerifySubField(expr, expr.Predicate, TypePatternUtility.IsBoolScalar());
        }
        catch (TypeInferenceInterruptException e)
        {
            SetCheckedType(expr, e.ReasonType);
            return e.ReasonType;
        }
        var type = TupleType.Void;
        SetCheckedType(expr, type);
        return type;
    }

    public override IRType VisitLeaf(BufferLoad expr)
    {
        try
        {
            VerifySubField(expr, expr.Buffer, TypePatternUtility.IsPointer());
            foreach (var i in Enumerable.Range(0, expr.Indices.Count))
            {
                VerifySubField(expr, expr.Indices[i], TypePatternUtility.IsIntegralScalar(), $"BufferLoad.Indices[{i}]");
            }
        }
        catch (TypeInferenceInterruptException e)
        {
            SetCheckedType(expr, e.ReasonType);
            return e.ReasonType;
        }

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
        try
        {
            VerifySubField(expr, expr.Buffer, TypePatternUtility.IsPointer());
            foreach (var i in Enumerable.Range(0, expr.Indices.Count))
            {
                VerifySubField(expr, expr.Indices[i], TypePatternUtility.IsIntegralScalar(), $"BufferStore.Indices[{i}]");
            }
            VerifySubField(expr, expr.Value, TypePatternUtility.IsScalar());
        }
        catch (TypeInferenceInterruptException e)
        {
            SetCheckedType(expr, e.ReasonType);
            return e.ReasonType;
        }

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
        try
        {
            VerifySubField(expr, expr.Condition, TypePatternUtility.IsBoolScalar());
            VerifySubField(expr, expr.Then, TypePatternUtility.IsUnit());
            VerifySubField(expr, expr.Else, TypePatternUtility.IsUnit());
        }
        catch (TypeInferenceInterruptException e)
        {
            SetCheckedType(expr, e.ReasonType);
            return e.ReasonType;
        }

        IRType type = TupleType.Void;
        SetCheckedType(expr, type);
        return type;
    }

    public override IRType Visit(Let expr)
    {
        if (!ExpressionMemo.TryGetValue(expr, out var result))
        {
            Visit(expr.Expression);
            if (expr.Var.TypeAnnotation is not AnyType)
            {
                // now we need custom visit the var.
                result = new InvalidType("The Let Bind Var Must Be Any Type!");
                SetCheckedType(expr.Var, result);
                ExpressionMemo.Add(expr.Var, result);

                SetCheckedType(expr, result);
            }
            else
            {
                // now change the var checkedtype
                SetCheckedType(expr.Var, expr.Expression.CheckedType!);
                ExpressionMemo[expr.Var] = expr.Expression.CheckedType!;

                Visit(expr.Body);
                result = VisitLeaf(expr);
            }
            ExpressionMemo.Add(expr, result);
        }
        return result;
    }

    /// <inheritdoc/>
    public override IRType VisitLeaf(Let expr)
    {
        try
        {
            if (expr.Var.CheckedType != expr.Expression.CheckedType)
                throw new TypeInferenceInterruptException(new InvalidType("Var Type != Expression Type"));
            VerifySubField(expr, expr.Body, TypePatternUtility.IsUnit());
        }
        catch (TypeInferenceInterruptException e)
        {
            SetCheckedType(expr, e.ReasonType);
            return e.ReasonType;
        }

        IRType type = TupleType.Void;
        SetCheckedType(expr, type);
        return type;
    }

    /// <inheritdoc/>
    public override IRType VisitLeaf(Nncase.TIR.Buffer expr)
    {
        IRType type = expr.ElemType;
        SetCheckedType(expr, type);
        return type;
    }

    public override IRType VisitLeaf(Nncase.TIR.BufferRegion expr)
    {
        try
        {
            VerifySubField(expr, expr.Buffer, TypePatternUtility.IsTensor());
            foreach (var r in expr.Region)
            {
                VerifySubField(expr, r.Start, TypePatternUtility.IsIntegralScalar());
                VerifySubField(expr, r.Stop, TypePatternUtility.IsIntegralScalar());
                VerifySubField(expr, r.Stop, TypePatternUtility.IsIntegralScalar());
            }
        }
        catch (TypeInferenceInterruptException e)
        {
            SetCheckedType(expr, e.ReasonType);
            return e.ReasonType;
        }
        // todo need infer the sub region shape/stride
        IRType type = expr.Buffer.ElemType;
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
        IsFullyInferenced &= type is not InvalidType;
    }
}
