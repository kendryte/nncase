// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.Quantization;
using Nncase.Transform;
using Nncase.Transform.Passes;
using Xunit;

namespace Nncase.Tests.CliTest;

public class UnitTestCli : TestFixture.UnitTestFixtrue
{
    [Fact]
    public async Task TestTfliteNoptq()
    {
        string target = "cpu";
        string input_format = "tflite";
        int dump_level = 4;
        string dump_dir = $"UnitTestCli_{input_format}_noptq";
        string options = $" -t {target} -i {input_format} --dump-level {dump_level} --dump-dir {dump_dir} ";
        string input_file = Path.Combine(GetSolutionDirectory(), "tests/models/conv.tflite");
        string output_file = Path.Combine(dump_dir, "test.kmodel");

        await Compile(options, input_file, output_file);
    }

    [Fact]
    public async Task TestTflitePtq()
    {
        string target = "cpu";
        string input_format = "tflite";
        int dump_level = 4;
        string dump_dir = $"UnitTestCli_{input_format}_ptq";
        var quant_type = "UInt8";
        var wquant_type = "Uint8";
        var model_quant_mode = "UsePTQ";
        var calib_method = "Random";
        string options = $" -t {target} -i {input_format} --dump-level {dump_level} --dump-dir {dump_dir} ";
        options += $"--quant-type {quant_type} --wquant-type {wquant_type} --model-quant-mode {model_quant_mode} --calib-method {calib_method}";
        string input_file = Path.Combine(GetSolutionDirectory(), "tests/models/conv.tflite");
        string output_file = Path.Combine(dump_dir, "test.kmodel");

        await Compile(options, input_file, output_file);
    }

    [Fact]
    public async Task TestOnnxNoptq()
    {
        string target = "cpu";
        string input_format = "onnx";
        int dump_level = 4;
        string dump_dir = $"UnitTestCli_{input_format}_noptq";
        string options = $" -t {target} -i {input_format} --dump-level {dump_level} --dump-dir {dump_dir} ";
        string input_file = Path.Combine(GetSolutionDirectory(), "tests/models/conv.onnx");
        string output_file = Path.Combine(dump_dir, "test.kmodel");

        await Compile(options, input_file, output_file);
    }

    [Fact]
    public async Task TestOnnxPtq()
    {
        string target = "cpu";
        string input_format = "onnx";
        int dump_level = 4;
        string dump_dir = $"UnitTestCli_{input_format}_ptq";
        var quant_type = "UInt8";
        var wquant_type = "Uint8";
        var model_quant_mode = "UsePTQ";
        var calib_method = "Random";
        string options = $" -t {target} -i {input_format} --dump-level {dump_level} --dump-dir {dump_dir} ";
        options += $"--quant-type {quant_type} --wquant-type {wquant_type} --model-quant-mode {model_quant_mode} --calib-method {calib_method}";
        string input_file = Path.Combine(GetSolutionDirectory(), "tests/models/conv.onnx");
        string output_file = Path.Combine(dump_dir, "test.kmodel");

        await Compile(options, input_file, output_file);
    }

    private async Task Compile(string options, string input_file, string output_file)
    {
        string nncase_cli = Environment.GetEnvironmentVariable("NNCASE_CLI");
        Assert.True(!String.IsNullOrEmpty(nncase_cli));

        // run
        var psi = new ProcessStartInfo();
        psi.FileName = nncase_cli;
        psi.Arguments = " compile " + options + " " + input_file + " " + output_file;
        psi.UseShellExecute = false;
        psi.CreateNoWindow = true;

        var process = System.Diagnostics.Process.Start(psi);
        process.WaitForExit();
        Assert.True(process.ExitCode == 0);

        // check
        var stream = File.OpenRead(output_file);
        var bytes = new byte[stream.Length];
        stream.Read(bytes);
        Assert.True(bytes.Length > 0);
        Assert.True(bytes[0] == 0x4c);
        Assert.True(bytes[1] == 0x44);
        Assert.True(bytes[2] == 0x4d);
        Assert.True(bytes[3] == 0x4b);
    }
}