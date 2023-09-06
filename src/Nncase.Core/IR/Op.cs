// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.IR;

/// <summary>
/// Parameter information.
/// </summary>
public sealed class ParameterInfo
{
    /// <summary>
    /// Initializes a new instance of the <see cref="ParameterInfo"/> class.
    /// ctor.
    /// </summary>
    /// <param name="ownerType">this op type.</param>
    /// <param name="index">param index.</param>
    /// <param name="name">param name.</param>
    public ParameterInfo(Type ownerType, int index, string name)
    {
        OwnerType = ownerType;
        Index = index;
        Name = name;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="ParameterInfo"/> class.
    /// ctor.
    /// </summary>
    /// <param name="ownerType">this op type.</param>
    /// <param name="index">param index.</param>
    /// <param name="name">param name.</param>
    /// <param name="pattern">the param condition.</param>
    public ParameterInfo(Type ownerType, int index, string name, TypePattern pattern)
        : this(ownerType, index, name)
    {
        Pattern = pattern;
    }

    /// <summary>
    /// Gets the parameter info ownertype.
    /// </summary>
    public Type OwnerType { get; }

    /// <summary>
    /// Gets parameter index num.
    /// </summary>
    public int Index { get; }

    /// <summary>
    /// Gets name.
    /// </summary>
    public string Name { get; }

    /// <summary>
    /// Gets this paramter's type condition.
    /// </summary>
    public TypePattern Pattern { get; } = IsIRType();

    /// <summary>
    /// Check current type by pattern.
    /// </summary>
    /// <returns> check success. </returns>
    public bool CheckType(IRType type) => Pattern.MatchLeaf(type);
}

/// <summary>
/// Operator expression.
/// we will Reflection the specific method to extent the function,
/// so your need impl the method as follows:
/// 1. Visit(ITypeInferenceContext context, IRType arg1, IRType arg2, ...)
/// </summary>
public abstract class Op : Expr
{
    private ParameterInfo[]? _parameters;

    public Op()
        : base(Array.Empty<Expr>())
    {
    }

    /// <summary>
    /// Gets get the parameters.
    /// </summary>
    public virtual IEnumerable<ParameterInfo> Parameters =>
        _parameters ??= (from p in GetType().GetFields(BindingFlags.Public | BindingFlags.Static)
                         where p.FieldType == typeof(ParameterInfo)
                         let param = (ParameterInfo)(p.GetValue(null) ?? throw new InvalidOperationException())
                         orderby param.Index
                         select param).ToArray();

    /// <summary>
    /// Gets a value indicating whether mark this op can be fold when input's are const.
    /// </summary>
    public virtual bool CanFoldConstCall => true;

    /// <summary>
    /// display the Op property for dump ir.
    /// </summary>
    /// <returns> property string. </returns>
    public virtual string DisplayProperty()
    {
        return string.Empty;
    }

    /// <inheritdoc/>
    public sealed override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context)
        => functor.VisitOp(this, context);

    public Op With() => this;
}

/// <summary>
/// Custom Op.
/// </summary>
public abstract class CustomOp : Op
{
    public abstract string RegisteredName { get; }

    /// <summary>
    /// Gets get the Current Custom module type.
    /// </summary>
    public abstract CodeGen.ModuleType ModuleType { get; }

    /// <summary>
    /// Serialize Fields Value.
    /// will used in stackvm runtime.
    /// </summary>
    public virtual byte[] SerializeFields()
    {
        return Array.Empty<byte>();
    }
}
