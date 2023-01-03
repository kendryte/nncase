// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.CommandLine;
using System.CommandLine.Invocation;
using System.IO;
using System.Linq;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Nncase.CodeGen;
using Nncase.Compiler;
using Nncase.IR;
using Nncase.Transform;

namespace Nncase.Cli.Commands;

internal enum QuantType
{
    UInt8,
    Int8,
    Int16,
}

/// <summary>
/// Compile command.
/// </summary>
public class Compile : Command
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
          description: "dump ir to .il, default is 0",
          getDefaultValue: () => 0));
        AddOption(new Option<string>(
          alias: "--dump-dir",
          description: "dump to directory, default is .",
          getDefaultValue: () => "."));
        AddOption(new Option<QuantType>(
          alias: "--quant-type",
          description: "quant type, default is uint8",
          getDefaultValue: () => QuantType.UInt8));
        AddOption(new Option<QuantType>(
          alias: "--wquant-type",
          description: "wquant type, default is uint8",
          getDefaultValue: () => QuantType.UInt8));
        AddOption(new Option<Quantization.ModelQuantMode>(
          alias: "--model-quant-mode",
          description: "model quant mode, default is NoQuant",
          getDefaultValue: () => Quantization.ModelQuantMode.NoQuant));
        AddOption(new Option<Quantization.CalibMethod>(
          alias: "--calib-method",
          description: "model quant options, default is Random",
          getDefaultValue: () => Quantization.CalibMethod.Random));

        Handler = CommandHandler.Create<CliCompileOptions, IHost>(Run);
    }

    private void Run(CliCompileOptions cliOptions, IHost host)
    {
        CompilerServices.Configure(host.Services);

        // 1. setup the options
        var compileOptions = new CompileOptions(
            InputFile: cliOptions.InputFile,
            InputFormat: cliOptions.InputFormat,
            DumpLevel: cliOptions.DumpLevel,
            DumpDir: cliOptions.DumpDir,
            QuantizeOptions: new()
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
            });

        // 2. import the model
        var target = CompilerServices.GetTarget(cliOptions.Target);
        using var compileSession = CompileSession.Create(target, compileOptions);
        var compiler = compileSession.Compiler;
        IRModule module;
        using (var model_stream = File.OpenRead(compileOptions.InputFile))
        {
            module = compiler.ImportModule(model_stream);
        }

        // 3. create the calib dataset
        if (compileOptions.QuantizeOptions.ModelQuantMode == Quantization.ModelQuantMode.UsePTQ)
        {
            if (compileOptions.QuantizeOptions.CalibrationMethod == Quantization.CalibMethod.Random)
            {
                compileOptions.QuantizeOptions.CalibrationDataset = new RandCalibrationDatasetProvider(((Function)module.Entry!).Parameters.ToArray());
            }
        }

        // 4. compile
        compiler.Compile();

        // 5. code gen
        using (var os = File.OpenWrite(cliOptions.OutputFile))
        {
            compiler.Gencode(os);
        }
    }
}

internal sealed class CliCompileOptions
{
    /// <inheritdoc/>
    public string InputFile { get; set; }

    /// <inheritdoc/>
    public string InputFormat { get; set; }

    /// <inheritdoc/>
    public string Target { get; set; }

    /// <inheritdoc/>
    public int DumpLevel { get; set; }

    /// <inheritdoc/>
    public string DumpDir { get; set; }

    /// <inheritdoc/>
    public QuantType QuantType { get; set; }

    /// <inheritdoc/>
    public QuantType WQuantType { get; set; }

    /// <inheritdoc/>
    public string OutputFile { get; set; }

    /// <inheritdoc/>
    public Quantization.ModelQuantMode ModelQuantMode { get; set; }

    /// <inheritdoc/>
    public Quantization.CalibMethod CalibMethod { get; set; }
}

internal sealed class RandCalibrationDatasetProvider : Quantization.ICalibrationDatasetProvider
{
    private const int CountValue = 5;

    private readonly IReadOnlyDictionary<Var, IValue>[] _samples;

    public RandCalibrationDatasetProvider(IEnumerable<Var> vars)
    {
        _samples = Enumerable.Range(0, CountValue).Select(i =>
          {
              var values = new Dictionary<Var, IValue>();
              foreach (var var in vars)
              {
                  CompilerServices.InferenceType(var);
                  var shape = var.CheckedShape.Select(d => d.IsUnknown ? 1 : d.FixedValue).ToArray();
                  var value = IR.F.Random.Normal(var.CheckedDataType, 0, 1, 0, shape).Evaluate();
                  values.Add(var, value);
              }

              return values;
          }).ToArray();
    }

    public int? Count => CountValue;

    public IAsyncEnumerable<IReadOnlyDictionary<Var, IValue>> Samples => _samples.ToAsyncEnumerable();
}
