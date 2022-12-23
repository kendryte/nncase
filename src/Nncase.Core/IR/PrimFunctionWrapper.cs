// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.TIR;

namespace Nncase.IR;

/// <summary>
/// CallPrimFunction expression.
/// </summary>
public sealed record PrimFunctionWrapper(string Name, PrimFunction Target, int ParametersCount) : BaseFunction(Name, StackVMModuleKind)
{
    private static int _globalFuncIndex;

    /// <summary>
    /// Initializes a new instance of the <see cref="PrimFunctionWrapper"/> class.
    /// </summary>
    /// <param name="target">Target.</param>
    /// <param name="parametersCount">Parameters count.</param>
    public PrimFunctionWrapper(PrimFunction target, int parametersCount)
        : this($"func_{_globalFuncIndex++}", target, parametersCount)
    {
    }

    /// <summary>
    /// Gets return type.
    /// </summary>
    public IRType? ReturnType
    {
        get
        {
            var outputParams = Target.Parameters.Skip(ParametersCount).ToList();
            return outputParams.Count == 1
                ? outputParams[0].CheckedType
                : new TupleType(outputParams.Select(x => x.CheckedType!));
        }
    }

    /// <inheritdoc/>
    public override IEnumerable<IRType?> ParameterTypes => Target.Parameters.Take(ParametersCount).Select(x => x.CheckedType);
}
