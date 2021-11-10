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
        /// Inference type (nothrow).
        /// </summary>
        /// <param name="context">Context.</param>
        /// <returns>Inferred type.</returns>
        internal IRType InferInvokeResultTypeNoThrow(ITypeInferenceContext context)
        {
            var thistype = this.GetType();
            var properties = thistype.GetFields(
              BindingFlags.Public |
              BindingFlags.Static);

            var typeinferFunc = thistype.GetMethod("InferInvokeResultType") ??
               throw new InvalidProgramException("The Ops Must Have `InferInvokeResultType` method!");
            var inferFuncParams = typeinferFunc.GetParameters();
            var inferTypedict = inferFuncParams.Skip(1).ToDictionary(p => p.Name ?? "_", p => p.ParameterType);
            var targetParams = new List<object> { context };
            foreach (var info in properties)
            {
                var paraminfo = (ParameterInfo)(info.GetValue(null)
                    ?? throw new InvalidProgramException($"Can't Get The ParameterInfo {info.Name}"));
                var targetType = inferTypedict[paraminfo.Name];
                var paramActualType = context.GetArgumentType(this, paraminfo).ThrowIfTypeInferenceInterrupt();
                targetParams.Add(Convert.ChangeType(paramActualType, targetType));
            }
            return ((IRType)(typeinferFunc.Invoke(this, targetParams.ToArray())
                  ?? throw new InvalidProgramException("The InferInvokeResultType Function must return IRType!"))).ThrowIfTypeInferenceInterrupt();

        }
    }
}
