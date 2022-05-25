// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.CodeGen;
using Nncase.CodeGen.StackVM;
using Nncase.IR;

namespace Nncase.Targets;

/// <summary>
/// Target for CPU.
/// </summary>
public class CPUTarget : ITarget
{
    /// <inheritdoc/>
    public string Kind => "cpu";

    /// <inheritdoc/>
    public IModuleBuilder CreateModuleBuilder(string moduleKind)
    {
        if (moduleKind == Callable.StackVMModuleKind)
        {
            return new StackVMModuleBuilder();
        }
        else
        {
            throw new NotSupportedException($"{moduleKind} module is not supported.");
        }
    }
}
