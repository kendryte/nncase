// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.CommandLine;
using System.CommandLine.Invocation;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Options;
using Nncase.CodeGen;
using Nncase.CodeGen.NTT;
using Nncase.IR;
using Nncase.Passes;
using Nncase.Passes.Rules.Neutral;
using Nncase.Passes.Rules.ShapeBucket;
using Nncase.Passes.Transforms;
using Nncase.Quantization;

namespace Nncase.Targets;

/// <summary>
/// Target for CUDA.
/// </summary>
public class CUDATarget : NTTTarget
{
    public const string Kind = "cuda";

    public CUDATarget()
    {
    }

    protected override INTTModuleCompiler NTTModuleCompiler { get; } = new CUDAModuleCompiler();
}
