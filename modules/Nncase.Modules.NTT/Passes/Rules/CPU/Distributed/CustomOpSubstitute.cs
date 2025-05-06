// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Nncase.IR;
using Nncase.Passes.Rules.NTT.CustomOp;
using Nncase.PatternMatch;
using Nncase.Targets;

namespace Nncase.Passes.Distributed;

/// <summary>
/// CustomOpSubstitute pass.
///
/// </summary>
public class CustomOpSubstitutePass : DataflowPass
{
    private readonly CompileOptions _compileOptions;

    private readonly CpuTargetOptions _cpuTargetOptions;

    public CustomOpSubstitutePass(CompileOptions compileOptions)
    {
        _compileOptions = compileOptions;

        _cpuTargetOptions = _compileOptions.TargetOptions is CpuTargetOptions options ? options : new CpuTargetOptions();

        if (Path.Exists(_cpuTargetOptions.CustomOpScheme) &&
        System.Text.Json.JsonSerializer.Deserialize<CustomOpScheme>(File.ReadAllText(_cpuTargetOptions.CustomOpScheme)) is CustomOpScheme customOpScheme)
        {
            CustomOpScheme = customOpScheme;
        }
        else
        {
            CustomOpScheme = null;
        }

        if (CustomOpScheme is not null)
        {
            Add<ToCustomUnary>(CustomOpScheme);
            Add<ToCustomMatmul>(CustomOpScheme);
        }
    }

    public CustomOpScheme? CustomOpScheme { get; }

    protected override Task<BaseFunction> RunCoreAsync(BaseFunction function, RunPassContext options)
    {
        return Task.FromResult((BaseFunction)CompilerServices.Rewrite(function, Rules, options));
    }
}
