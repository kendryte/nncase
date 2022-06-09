// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.CodeGen;
using Nncase.CodeGen.K210;
using Nncase.IR;
using Nncase.Transform;

namespace Nncase.Targets;

/// <summary>
/// Target for K210.
/// </summary>
public class K210Target : ITarget
{
    /// <inheritdoc/>
    public string Kind => "k210";

    /// <inheritdoc/>
    public void RegisterTargetDependentPass(PassManager passManager, ICompileOptions options)
    {
    }

    /// <inheritdoc/>
    public void RegisterQuantizePass(PassManager passManager)
    {
    }

    /// <inheritdoc/>
    public void RegisterTargetDependentAfterQuantPass(PassManager passManager)
    {
    }

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
