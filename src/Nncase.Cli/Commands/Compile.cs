﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.CommandLine;
using System.CommandLine.Invocation;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Nncase.CodeGen;
using Nncase.Compiler;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.Passes;
using Nncase.Quantization;

namespace Nncase.Cli.Commands;

internal enum QuantType
{
    UInt8,
    Int8,
    Int16,
}

internal enum DatasetFormat
{
    Image,
    Raw,
    Pytest,
    Random,
}

/// <summary>
/// Compile command.
/// </summary>
public sealed class Compile : Command
{
    /// <summary>
    /// Initializes a new instance of the <see cref="Compile"/> class.
    /// </summary>
    public Compile()
        : base("compile")
    {
        AddArgument(new Argument("input-file"));
        AddArgument(new Argument("output-file"));
        AddOption(new Option<string>(
          aliases: new string[] { "-t", "--target" },
          description: "target architecture, e.g. cpu, k210"));
        AddOption(new Option<string>(
          aliases: new[] { "-i", "--input-format" },
          description: "input format, e.g. tflite",
          getDefaultValue: () => "tflite"));
        AddOption(new Option<int>(
          alias: "--dump-level",
          description: $"dump ir to .il, default is {0}",
          getDefaultValue: () => 0));
        AddOption(new Option<string>(
          alias: "--dump-dir",
          description: "dump to directory, default is .",
          getDefaultValue: () => "."));
        AddOption(new Option<QuantType>(
          alias: "--quant-type",
          description: $"quant type, default is {QuantType.UInt8}",
          getDefaultValue: () => QuantType.UInt8));
        AddOption(new Option<QuantType>(
          alias: "--wquant-type",
          description: $"wquant type, default is {QuantType.UInt8}",
          getDefaultValue: () => QuantType.UInt8));
        AddOption(new Option<string>(
          alias: "--dataset",
          description: $"calibration dataset, used in post quantization, default is empty",
          getDefaultValue: () => string.Empty));
        AddOption(new Option<DatasetFormat>(
          alias: "--dataset-format",
          description: $"datset format: e.g. Image|Raw|Pytest",
          getDefaultValue: () => DatasetFormat.Raw));
        AddOption(new Option<Quantization.ModelQuantMode>(
          alias: "--model-quant-mode",
          description: $"model quant mode, default is {Quantization.ModelQuantMode.NoQuant}",
          getDefaultValue: () => Quantization.ModelQuantMode.NoQuant));
        AddOption(new Option<Quantization.CalibMethod>(
          alias: "--calib-method",
          description: $"model quant options, default is {Quantization.CalibMethod.Kld}",
          getDefaultValue: () => Quantization.CalibMethod.Kld));

        Handler = CommandHandler.Create<CliCompileOptions, IHost>(RunAsync);
    }

    private static DumpFlags DumpLevelToFlags(int dumpLevel)
    {
        return dumpLevel switch
        {
            0 => DumpFlags.None,
            1 => DumpLevelToFlags(0) | DumpFlags.Compile,
            2 => DumpLevelToFlags(1) | DumpFlags.PassIR,
            3 => DumpLevelToFlags(2) | DumpFlags.Rewrite,
            4 => DumpLevelToFlags(3) | DumpFlags.EGraphCost,
            5 => DumpLevelToFlags(4) | DumpFlags.Evaluator,
            6 => DumpLevelToFlags(5) | DumpFlags.Calibration,
            7 => DumpLevelToFlags(6) | DumpFlags.Tiling,
            8 => DumpLevelToFlags(7) | DumpFlags.Schedule,
            >= 9 => DumpLevelToFlags(8) | DumpFlags.CodeGen,
            _ => throw new ArgumentOutOfRangeException(nameof(dumpLevel)),
        };
    }

    private async Task RunAsync(CliCompileOptions cliOptions, IHost host)
    {
        CompilerServices.Configure(host.Services);

        // 1. setup the options
        var compileOptions = new CompileOptions
        {
            InputFile = cliOptions.InputFile,
            InputFormat = cliOptions.InputFormat,
            DumpFlags = DumpLevelToFlags(cliOptions.DumpLevel),
            DumpDir = cliOptions.DumpDir,
            QuantizeOptions = new()
            {
                CalibrationMethod = cliOptions.CalibMethod,
                QuantType = cliOptions.QuantType switch
                {
                    QuantType.UInt8 => DataTypes.UInt8,
                    QuantType.Int8 => DataTypes.Int8,
                    QuantType.Int16 => DataTypes.Int16,
                    _ => throw new ArgumentException("Invalid quant type"),
                },
                WQuantType = cliOptions.WQuantType switch
                {
                    QuantType.UInt8 => DataTypes.UInt8,
                    QuantType.Int8 => DataTypes.Int8,
                    QuantType.Int16 => DataTypes.Int16,
                    _ => throw new ArgumentException("Invalid weights quant type"),
                },
                ModelQuantMode = cliOptions.ModelQuantMode,
            },
        };

        // 2. import the model
        var target = CompilerServices.GetTarget(cliOptions.Target);
        using var compileSession = CompileSession.Create(target, compileOptions);
        var compiler = compileSession.Compiler;
        IRModule module;
        using (var model_stream = File.OpenRead(compileOptions.InputFile))
        {
            module = await compiler.ImportModuleAsync(model_stream);
        }

        // 3. create the calib dataset
        if (compileOptions.QuantizeOptions.ModelQuantMode == Quantization.ModelQuantMode.UsePTQ)
        {
            if (cliOptions.DatasetFormat == DatasetFormat.Random)
            {
                compileOptions.QuantizeOptions.CalibrationDataset = new RandomCalibrationDatasetProvider(((Function)module.Entry!).Parameters.ToArray(), 5);
            }
            else if (cliOptions.DatasetFormat == DatasetFormat.Pytest)
            {
                compileOptions.QuantizeOptions.CalibrationDataset = new PytestCalibrationDatasetProvider(((Function)module.Entry!).Parameters.ToArray(), cliOptions.Dataset);
            }
            else
            {
                throw new NotSupportedException(cliOptions.DatasetFormat.ToString());
            }
        }

        // 4. compile
        await compiler.CompileAsync();

        // 5. code gen
        using (var os = File.OpenWrite(cliOptions.OutputFile))
        {
            compiler.Gencode(os);
        }
    }
}

// Validate null in command line parser.
#pragma warning disable CS8618

internal sealed class CliCompileOptions
{
    public string InputFile { get; set; }

    public string InputFormat { get; set; }

    public string Target { get; set; }

    public int DumpLevel { get; set; }

    public string DumpDir { get; set; }

    public QuantType QuantType { get; set; }

    public QuantType WQuantType { get; set; }

    public string OutputFile { get; set; }

    public ModelQuantMode ModelQuantMode { get; set; }

    public CalibMethod CalibMethod { get; set; }

    public string Dataset { get; set; }

    public DatasetFormat DatasetFormat { get; set; }
}

#pragma warning restore CS8618
