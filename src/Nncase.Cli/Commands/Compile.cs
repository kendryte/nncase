// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.CommandLine;
using System.CommandLine.Invocation;
using System.IO;
using System.Linq;
using Autofac;
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
    public QuantMode QuantMode { get; set; }

    /// <inheritdoc/>
    public string OutputFile { get; set; }

    /// <inheritdoc/>
    public Quantization.ModelQuantMode ModelQuantMode { get; set; }

    /// <inheritdoc/>
    public Quantization.CalibMethod CalibMethod { get; set; }
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
          description: "target architecture, e.g. cpu, k210")
        );
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
        AddOption(new Option<QuantMode>(
          alias: "--quant-mode",
          description: "quant model, default is UnsignedMode",
          getDefaultValue: () => QuantMode.UnsignedMode));
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
        var provider = host.Services.GetRequiredService<ICompilerServicesProvider>();
        CompilerServices.Configure(provider);

        // 1. setup the options
        Quantization.QuantizeOptions quant_options = new() { CalibrationMethod = cliOptions.CalibMethod };
        var compileOptions = new CompileOptions()
        {
            InputFile = cliOptions.InputFile,
            OutputFile = cliOptions.OutputFile,
            Target = cliOptions.Target,
            InputFormat = cliOptions.InputFormat,
            DumpLevel = cliOptions.DumpLevel,
            DumpDir = cliOptions.DumpDir,
            QuantType = cliOptions.QuantType switch
            {
                QuantType.UInt8 => DataTypes.UInt8,
                QuantType.Int8 => DataTypes.Int8,
                QuantType.Int16 => DataTypes.Int16,
                _ => throw new ArgumentOutOfRangeException()
            },
            QuantMode = cliOptions.QuantMode,
            ModelQuantMode = cliOptions.ModelQuantMode,
            // todo add the quant options parser
            QuantizeOptions = quant_options,
        };

        // 2. import the model
        Compiler.Compiler.UpdateCompileOptions(compileOptions);
        var compiler = new Compiler.Compiler();
        using (var model_stream = File.OpenRead(CompilerServices.CompileOptions.InputFile))
        {
            compiler.ImportModule(model_stream);
        }

        // 3. create the calib dataset
        if (compileOptions.ModelQuantMode == Quantization.ModelQuantMode.UsePTQ)
        {
            if (quant_options.CalibrationMethod == Quantization.CalibMethod.Random)
            {
                quant_options.CalibrationDataset = new RandCalibrationDatasetProvider(((Function)compiler.Module.Entry!).Parameters.ToArray());
            }
        }

        // 4. compile
        compiler.Compile();

        // 5. code gen
        var bytes = compiler.Gencode();
        using (var os = File.OpenWrite(CompilerServices.CompileOptions.OutputFile))
        {
            os.Write(bytes);
        }
    }


}


internal sealed class RandCalibrationDatasetProvider : Quantization.ICalibrationDatasetProvider
{

    private const int count = 5;
    public int? Count => count;

    public IAsyncEnumerable<IReadOnlyDictionary<Var, IValue>> Samples { get; }

    public RandCalibrationDatasetProvider(IEnumerable<Var> vars)
    {
        Samples = Enumerable.Range(0, count).Select(i =>
        {
            var values = new Dictionary<Var, IValue>();
            foreach (var var in vars)
            {
                CompilerServices.InferenceType(var);
                var shape = var.CheckedShape.Select(d => d.IsUnknown ? 1 : d.FixedValue).ToArray();
                var value = Value.FromTensor(IR.F.Random.Normal(var.CheckedDataType, 0, 1, 0, shape).Evaluate().AsTensor());
                values.Add(var, value);
            }
            return values;
        }).ToAsyncEnumerable();
    }
}
