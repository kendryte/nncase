// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.CommandLine;
using System.CommandLine.Binding;
using System.CommandLine.Invocation;
using System.IO;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;

namespace CApiGen;

public sealed class Program
{
    internal static async Task Main(string[] args)
    {
        await Extract(string.Empty);
    }

    internal static async Task Extract(string inputFile, [CallerFilePath] string? callerFilePath = null)
    {
        var extractor = new CommandExtractor();
        var optionType = typeof(Nncase.Targets.CpuTargetOptions);
        extractor.Extract(optionType);

        var directory = Path.GetDirectoryName(callerFilePath);
        var command = await extractor.RenderAsync("Templates.Command");
        File.WriteAllText(Path.Combine($"{directory}/out", "Command.cs"), command);

        var capi = await extractor.RenderAsync("Templates.CApi");
        File.WriteAllText(Path.Combine($"{directory}/out", "CApi.cs"), capi);

        var compiler = await extractor.RenderAsync("Templates.Compiler");
        File.WriteAllText(Path.Combine($"{directory}/out", "compiler.h"), compiler);

        var pybindh = await extractor.RenderAsync("Templates.PyBind");
        File.WriteAllText(Path.Combine($"{directory}/out", "ffi.cpp"), pybindh);

        var python = await extractor.RenderAsync("Templates.Python");
        File.WriteAllText(Path.Combine($"{directory}/out", "_nncase.pyi"), python);
    }
}
