// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Nncase.CodeGen;
using Nncase.CodeGen.NTT;
using Nncase.IR;
using Nncase.Passes;

namespace Nncase.Targets;

public class NTTModuleCompiler : IModuleCompiler
{
    public string ModuleKind => CPUTarget.Kind;

    public MaskVectorStyle MaskVectorStyle => RuntimeInformation.ProcessArchitecture switch
    {
        Architecture.X64 or Architecture.Arm64 => MaskVectorStyle.Fat,
        _ => throw new NotSupportedException($"Unsupported architecture: {RuntimeInformation.ProcessArchitecture}"),
    };

    public IModuleBuilder CreateModuleBuilder(CompileOptions options) => new NTTModuleBuilder(options);

    public bool IsSupportedCall(Call call, CompileOptions options)
    {
        return call.Target switch
        {
            Op op => PassUtility.IsCpuSupported(op, call, call.Arguments, ModuleKind),
            _ => false,
        };
    }
}
