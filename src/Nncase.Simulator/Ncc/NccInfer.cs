using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;

namespace Nncase.Simulator.Ncc;

public class InferEngine
{

    string modelPath;
    string inputPath = string.Empty;
    string outputPath = string.Empty;

    public InferEngine(string model_path)
    {
        modelPath = model_path;
        inputPath = Path.Combine(Path.GetDirectoryName(modelPath) ?? "", "infer_input", "0.bin");
        outputPath = Path.Combine(Path.GetDirectoryName(modelPath) ?? "", "infer_output");
    }

    public void InputTensor(int index, Tensor tensor)
    {
        if (index != 0)
            throw new NotSupportedException("Only Support 1 Input!");
        tensor.ToFile(inputPath);
    }

    public Tensor OutputTensor(int index, DataType dataType, int[] shape)
    {
        if (index != 0)
            throw new NotSupportedException("Only Support 1 Output!");
        var bytes = File.ReadAllBytes(Path.Combine(outputPath, $"{index}.bin"));
        return Tensor.FromBytes(dataType,bytes,shape);
    }

    static string GetRid()
    {
        string os = string.Empty;
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
        {
            os = "linux";
        }
        else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
        {
            os = "osx";
        }
        else if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            os = "win";
        }

        var arch = string.Empty;
        arch = RuntimeInformation.ProcessArchitecture switch
        {
            Architecture.Arm64 => "arm64",
            Architecture.X64 => "x64",
            _ => throw new NotSupportedException()
        };

        return $"{os}-{arch}";
    }

    /// <summary>
    /// infer the model
    /// </summary>
    /// <exception cref="InvalidOperationException"></exception>
    public void Run()
    {
        var rid = GetRid();

        var errMsg = new StringBuilder();
        var logMsg = new StringBuilder();
        using var errWriter = new StringWriter(errMsg);
        using var logWriter = new StringWriter(logMsg);
        using var proc = new Process();
        proc.StartInfo.FileName = Path.Combine("runtimes", GetRid(), "native", "ncc");
        proc.StartInfo.Arguments = $"infer {modelPath} {outputPath} --dataset {inputPath} --dataset-format raw --wait-key"; proc.StartInfo.UseShellExecute = false;
        proc.StartInfo.RedirectStandardInput = true;
        proc.StartInfo.RedirectStandardError = true;
        proc.StartInfo.RedirectStandardOutput = true;
        if (Environment.GetEnvironmentVariable("DYLD_LIBRARY_PATH") is null)
        {
            proc.StartInfo.EnvironmentVariables["DYLD_LIBRARY_PATH"] = Environment.GetEnvironmentVariable("LD_LIBRARY_PATH") ?? string.Empty;
        }
        proc.ErrorDataReceived += (sender, e) => errWriter.WriteLine(e.Data);
        proc.OutputDataReceived += (sender, msg) => logWriter.Write(msg);
        proc.Start();
        proc.BeginOutputReadLine();
        proc.BeginErrorReadLine();
        proc.StandardInput.WriteLine("Start!");
        proc.StandardInput.Close();
        proc.WaitForExit();
        if (proc.ExitCode != 0)
        {
            throw new InvalidOperationException(errMsg.ToString());
        }
    }
}

