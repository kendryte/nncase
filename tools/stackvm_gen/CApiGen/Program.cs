// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.CommandLine;
using System.CommandLine.Binding;
using System.CommandLine.Invocation;
using System.IO;
using System.Threading.Tasks;

namespace CApiGen;

public sealed class Program
{
    internal static async Task Main(string[] args)
    {
        await Extract(string.Empty);
    }

    internal static async Task Extract(string inputFile)
    {
        var extractor = new CommandExtractor();
        var optionType = typeof(Nncase.Targets.CpuTargetOptions);
        extractor.Extract(optionType);

        var command = await extractor.RenderAsync("Templates.Command");
        File.WriteAllText(Path.Combine("/Users/lisa/Documents/nncase/tools/stackvm_gen/CApiGen", "Command.cs"), command);
    }
}
