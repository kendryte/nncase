// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.TIR;

/// <summary>
/// PrimFunction expression.
/// </summary>
public sealed record PrimFunction(string Name, string ModuleKind, Sequential Body, IRArray<Buffer> Parameters) : Callable(Name, ModuleKind), IFunction
{
    private static int _globalFuncIndex = 0;

    /// <summary>
    /// Initializes a new instance of the <see cref="PrimFunction"/> class.
    /// </summary>
    /// <param name="parameters">Parameters.</param>
    /// <param name="body">Body.</param>
    public PrimFunction(string moduleKind, Sequential body, IRArray<Buffer> parameters)
        : this($"primfunc_{_globalFuncIndex++}", moduleKind, body, parameters)
    {
    }

    /// <summary>
    /// build function.
    /// </summary>
    /// <param name="body"></param>
    /// <param name="parameters"></param>
    public PrimFunction(string moduleKind, Sequential body, params Buffer[] parameters)
        : this($"primfunc_{_globalFuncIndex++}", moduleKind, body, new(parameters))
    {
    }

    IEnumerable<IRType?> IFunction.ParameterTypes => Parameters.Select(x => x.CheckedType);
}
