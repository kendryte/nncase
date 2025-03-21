// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using Nncase.CodeGen;
using Nncase.CodeGen.CPU;
using Nncase.IR;
using Nncase.Passes;

namespace Nncase.Targets;

public class CPUModuleCompiler : IModuleCompiler
{
    public string ModuleKind => CPUTarget.Kind;

    public IModuleBuilder CreateModuleBuilder(CompileOptions options) => new CPUModuleBuilder(options);

    public bool IsSupportedCall(Call call, CompileOptions options)
    {
        return call.Target switch
        {
            Op op => PassUtility.IsCpuSupported(op, call, call.Arguments, ModuleKind),
            _ => false,
        };
    }
}
