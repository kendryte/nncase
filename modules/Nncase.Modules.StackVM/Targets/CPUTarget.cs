// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.Options;
using Nncase.CodeGen;
using Nncase.CodeGen.StackVM;
using Nncase.IR;
using Nncase.Transform;

namespace Nncase.Targets;

/// <summary>
/// Target for CPU.
/// </summary>
public class CPUTarget : ITarget
{
    /// <inheritdoc/>
    public string Kind => "cpu";

    /// <inheritdoc/>
    public CompileOptions CompileOptions { get; }

    public CPUTarget(IOptions<CompileOptions> compile_options)
    {
        CompileOptions = compile_options.Value;
    }

    public void RegisterTargetDependentPass(PassManager passManager, CompileOptions options)
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
