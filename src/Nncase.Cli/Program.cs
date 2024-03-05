// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.CommandLine;
using System.CommandLine.Builder;
using System.CommandLine.Hosting;
using System.CommandLine.Parsing;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Hosting;
using Nncase.Hosting;

namespace Nncase.Cli;

internal partial class Program
{
    public static async Task<int> Main(string[] args)
    {
        return await ConfigureCommandLine()
            .UseHost(ConfigureHost)
            .UseDefaults()
            .Build().InvokeAsync(args);
    }

    private static async Task RunAsync(string targetKind, CompileOptions compileOptions, DatasetFormat datasetFormat, string dataset, string outputFile, IHost host)
    {
        CompilerServices.Configure(host.Services);

        // 2. import the model
        var target = CompilerServices.GetTarget(targetKind);
        using var compileSession = CompileSession.Create(target, compileOptions);
        var compiler = compileSession.Compiler;
        IR.IRModule module = await compiler.ImportModuleAsync(Path.GetExtension(compileOptions.InputFile).Trim('.'), compileOptions.InputFile);

        // 3. create the calib dataset
        if (compileOptions.QuantizeOptions.ModelQuantMode == Quantization.ModelQuantMode.UsePTQ)
        {
            if (datasetFormat == DatasetFormat.Random)
            {
                compileOptions.QuantizeOptions.CalibrationDataset = new Quantization.RandomCalibrationDatasetProvider(((Nncase.IR.Function)module.Entry!).Parameters.ToArray(), 5);
            }
            else if (datasetFormat == DatasetFormat.Pytest)
            {
                compileOptions.QuantizeOptions.CalibrationDataset = new Quantization.PytestCalibrationDatasetProvider(((IR.Function)module.Entry!).Parameters.ToArray(), dataset);
            }
            else
            {
                throw new NotSupportedException(datasetFormat.ToString());
            }
        }

        // 4. compile
        await compiler.CompileAsync();

        // 5. code gen
        using (var os = File.OpenWrite(outputFile))
        {
            compiler.Gencode(os);
        }
    }

    private static CommandLineBuilder ConfigureCommandLine()
    {
        var compile = new CompileCommand();
        foreach (var target in LoadTargets())
        {
            var (targetCmd, targetParser) = target.RegisterCommandAndParser();
            Action<System.CommandLine.Invocation.InvocationContext> targetHandler = async (System.CommandLine.Invocation.InvocationContext context) =>
            {
                var options = ParseCompileOptions(context, compile);
                options.TargetCompileOptions = targetParser(context, targetCmd);
                await RunAsync(targetCmd.Name, options, context.ParseResult.GetValueForOption(compile.DatasetFormat), context.ParseResult.GetValueForOption(compile.Dataset)!, context.ParseResult.GetValueForArgument(compile.OutputFile), context.GetHost());
            };
            targetCmd.SetHandler(targetHandler);
            compile.AddCommand(targetCmd);
        }

        return new CommandLineBuilder(new RootCommand() { compile });
    }

    private static CompileOptions ParseCompileOptions(System.CommandLine.Invocation.InvocationContext context, CompileCommand compilecmd)
    {
        // 1. setup the options
        var compileOptions = new CompileOptions
        {
            InputFile = context.ParseResult.GetValueForArgument(compilecmd.InputFile),
            InputFormat = context.ParseResult.GetValueForOption(compilecmd.InputFormat)!,
            DumpFlags = context.ParseResult.GetValueForOption(compilecmd.DumpFlags)!.Aggregate(Diagnostics.DumpFlags.None, (a, b) => a | b),
            DumpDir = context.ParseResult.GetValueForOption(compilecmd.DumpDir)!,
            PreProcess = context.ParseResult.GetValueForOption(compilecmd.PreProcess)!,
            InputLayout = context.ParseResult.GetValueForOption(compilecmd.InputLayout)!,
            OutputLayout = context.ParseResult.GetValueForOption(compilecmd.OutputLayout)!,
            InputType = context.ParseResult.GetValueForOption(compilecmd.InputType)!,
            InputShape = context.ParseResult.GetValueForOption(compilecmd.InputShape)!.ToArray(),
            InputRange = context.ParseResult.GetValueForOption(compilecmd.InputRange)!.ToArray(),
            SwapRB = context.ParseResult.GetValueForOption(compilecmd.SwapRB)!,
            LetterBoxValue = context.ParseResult.GetValueForOption(compilecmd.LetterBoxValue)!,
            Mean = context.ParseResult.GetValueForOption(compilecmd.Mean)!.ToArray(),
            Std = context.ParseResult.GetValueForOption(compilecmd.Std)!.ToArray(),
            ModelLayout = context.ParseResult.GetValueForOption(compilecmd.ModelLayout)!,
            QuantizeOptions = new()
            {
                CalibrationMethod = context.ParseResult.GetValueForOption(compilecmd.CalibMethod),
                QuantType = context.ParseResult.GetValueForOption(compilecmd.QuantType) switch
                {
                    QuantType.UInt8 => DataTypes.UInt8,
                    QuantType.Int8 => DataTypes.Int8,
                    QuantType.Int16 => DataTypes.Int16,
                    _ => throw new ArgumentException("Invalid quant type"),
                },
                WQuantType = context.ParseResult.GetValueForOption(compilecmd.WQuantType) switch
                {
                    QuantType.UInt8 => DataTypes.UInt8,
                    QuantType.Int8 => DataTypes.Int8,
                    QuantType.Int16 => DataTypes.Int16,
                    _ => throw new ArgumentException("Invalid weights quant type"),
                },
                ModelQuantMode = context.ParseResult.GetValueForOption(compilecmd.ModelQuantMode),
            },
        };

        foreach (var item in context.ParseResult.GetValueForOption(compilecmd.FixedVars)!)
        {
            compileOptions.ShapeBucketOptions.FixVarMap.Add(item.Name, item.Value);
        }

        return compileOptions;
    }

    private static IReadOnlyList<ITarget> LoadTargets()
    {
        var loadContext = System.Runtime.Loader.AssemblyLoadContext.Default;
        var pluginAsms = PluginLoader.GetPluginsSearchDirectories(PluginLoader.PluginPathEnvName, null).
            Select(PluginLoader.GetPluginAssemblies).
            SelectMany(x => x).
            DistinctBy(Path.GetFileName).
            Select(x => PluginLoader.LoadPluginAssembly(x, loadContext)).
            Distinct().
            ToList();
        pluginAsms.AddRange(new[] { Path.GetDirectoryName(typeof(Program).Assembly.Location)! }.
            Select(basePath =>
            {
                if (Directory.Exists(basePath))
                {
                    return (from filePath in Directory.GetFiles(basePath, PluginLoader.ModulesDllPattern, SearchOption.AllDirectories)
                            where PluginLoader.IsLoadableAssembly(filePath)
                            select filePath).Distinct();
                }
                else
                {
                    return Array.Empty<string>();
                }
            }).
            SelectMany(x => x).
            DistinctBy(Path.GetFileName).
            Select(x => PluginLoader.LoadPluginAssembly(x, loadContext)).
            Distinct());
        var targets = (from asm in pluginAsms
                       from t in asm.ExportedTypes
                       where t.IsClass
                       && t.IsAssignableTo(typeof(ITarget))
                       let ctor = t.GetConstructor(Type.EmptyTypes)
                       where ctor != null
                       select (ITarget)ctor.Invoke(null)).ToList();
        return targets;
    }

    private static void ConfigureHost(IHostBuilder hostBuilder)
    {
        hostBuilder.ConfigureAppConfiguration(ConfigureAppConfiguration)
            .UseConsoleLifetime()
            .ConfigureCompiler();
    }

    private static void ConfigureAppConfiguration(HostBuilderContext context, IConfigurationBuilder builder)
    {
        var baseDirectory = Path.GetDirectoryName(typeof(Program).Assembly.Location);
        builder.SetBasePath(baseDirectory!)
            .AddJsonFile("config.json", true, false);
    }
}
