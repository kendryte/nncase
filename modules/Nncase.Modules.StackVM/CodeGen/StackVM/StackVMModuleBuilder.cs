// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.Runtime.StackVM;

namespace Nncase.CodeGen.StackVM;

/// <summary>
/// StackVM module builder.
/// </summary>
public class StackVMModuleBuilder : IModuleBuilder
{
    /// <inheritdoc/>
    public string ModuleKind => StackVMRTModule.Kind;

    /// <inheritdoc/>
    public IRTModule Build(IReadOnlyList<Callable> functions)
    {
        throw new NotImplementedException();
    }
}
