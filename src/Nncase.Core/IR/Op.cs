// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using static Nncase.IR.Utility;

namespace Nncase.IR
{
    /// <summary>
    /// Parameter information.
    /// </summary>
    public sealed class ParameterInfo
    {
        /// <summary>
        /// the parameter info ownertype.
        /// </summary>
        public Type OwnerType { get; }

        /// <summary>
        /// parameter index num.
        /// </summary>
        public int Index { get; }

        /// <summary>
        /// name.
        /// </summary>
        public string Name { get; }

        /// <summary>
        /// this paramter's type condition.
        /// </summary>
        public TypePattern Pattern { get; } = IsIRType();

        /// <summary>
        /// Check current type by pattern.
        /// </summary>
        /// <param name="type"></param>
        /// <returns> check success. </returns>
        public bool CheckType(IRType type) => Pattern.MatchLeaf(type);

        public ParameterInfo(Type ownerType, int index, string name)
        {
            OwnerType = ownerType;
            Index = index;
            Name = name;
        }

        public ParameterInfo(Type ownerType, int index, string name, TypePattern pattern) :
          this(ownerType, index, name)
        {
            Pattern = pattern;
        }
    }

    /// <summary>
    /// Operator expression.
    /// we will Reflection the specific method to extent the function,
    /// so your need impl the method as follows:
    /// 1. InferInvokeResultType(ITypeInferenceContext context, IRType arg1, IRType arg2, ...)
    /// </summary>
    public abstract record Op() : Expr
    {
        private ParameterInfo[]? _parameters;

        public IEnumerable<ParameterInfo> Parameters =>
            _parameters ??= (from p in this.GetType().GetProperties(BindingFlags.Public | BindingFlags.Static)
                             where p.PropertyType == typeof(ParameterInfo)
                             let param = (ParameterInfo)(p.GetValue(null) ?? throw new InvalidOperationException())
                             orderby param.Index
                             select param).ToArray();

        public virtual bool Equals(Op? other)
        {
            return !(other is null) && EqualityContract == other.EqualityContract;
        }

        public override int GetHashCode()
        {
            return _hashcode ??= EqualityComparer<Type>.Default.GetHashCode(EqualityContract);
        }
    }
}
