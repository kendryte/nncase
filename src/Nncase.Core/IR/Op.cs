// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.IR
{
    /// <summary>
    /// Parameter information.
    /// </summary>
    public sealed class ParameterInfo
    {
        public Type OwnerType { get; }

        public int Index { get; }

        public string Name { get; }

        public ParameterInfo(Type ownerType, int index, string name)
        {
            OwnerType = ownerType;
            Index = index;
            Name = name;
        }
    }

    /// <summary>
    /// Operator expression.
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

        /// <summary>
        /// Inference type.
        /// </summary>
        /// <param name="context">Context.</param>
        /// <returns>Inferred type.</returns>
        public abstract IRType InferInvokeResultType(ITypeInferenceContext context);

        /// <summary>
        /// Inference type (nothrow).
        /// </summary>
        /// <param name="context">Context.</param>
        /// <returns>Inferred type.</returns>
        internal IRType InferInvokeResultTypeNoThrow(ITypeInferenceContext context)
        {
            try
            {
                return InferInvokeResultType(context);
            }
            catch (TypeInferenceInterruptException ex)
            {
                return ex.Type;
            }
            catch (Exception ex)
            {
                return new InvalidType(ex.Message);
            }
        }
    }
}
