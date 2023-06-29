// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections;
using System.Diagnostics;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.TIR;
using Nncase.Utilities;
using Xunit;
using static Nncase.Tests.DumpPathExtractor;

namespace Nncase.Tests;

/// <summary>
/// Value with path.
/// </summary>
/// <param name="Value">Value.</param>
/// <param name="Path">Path.</param>
public record OriginValue(IValue Value, string Path)
{
    public string FileName => System.IO.Path.GetFileName(Path);

    public OriginTensor[] AsTensors() => Value.AsTensors().Select(t => new OriginTensor(t, Path)).ToArray();

    public OriginTensor AsTensor() => new OriginTensor(Value.AsTensor(), Path);
}

/// <summary>
/// Tensor with path.
/// </summary>
/// <param name="Tensor">Tensor.</param>
/// <param name="Path">Path.</param>
public record OriginTensor(Tensor Tensor, string Path) : OriginValue(Nncase.Value.FromTensor(Tensor), Path);

/// <summary>
/// Used for extract info from file name.
/// </summary>
public static class DumpPathExtractor
{
    // todo: rename
    public static char Separator => '$';

    public static int GetCount(string file) => int.Parse(file.Split(Separator).First());

    public static string GetOpName(string file) => file.Split(Separator)[1];

    public static string GetParamName(string file) =>
        IsParamFile(file)
            ? file.Split(Separator).Last()
            : throw new InvalidOperationException("file is not param");

    public static bool IsResultFile(string file) => file.Count(c => c == Separator) == 1;

    public static bool IsParamFile(string file) => file.Count(c => c == Separator) == 2;

    // used for transformer
    public static bool DynamicMatmulOnlyExtract(string fileName)
    {
        return fileName.Contains("mat", StringComparison.OrdinalIgnoreCase) && fileName.EndsWith("mul", StringComparison.OrdinalIgnoreCase);
    }
}

public class TextDataExtractor
{
    // FileNameFormat
    // input: (\d+)*$[a-z]*
    // param: (\d+)*$[a-z]*$[a-z]*
    public int GetDumpFileNum(string filePath)
    {
        var fileName = Path.GetFileName(filePath);
        if (fileName.Contains("out_shape_list", StringComparison.Ordinal))
        {
            return -1;
        }

        var match = System.Text.RegularExpressions.Regex
            .Match(fileName, @"(\d+)*");
        return int.Parse(match.Groups[0].Value);
    }

    public int FileNumSorter(string x, string y)
    {
        var a = GetDumpFileNum(x);
        var b = GetDumpFileNum(y);
        return a.CompareTo(b);
    }

    public List<string> GetFilesByOrdered(string dir)
    {
        var fs = Directory.GetFiles(dir).Where(x => FileFilter(Path.GetFileName(x))).ToList();
        fs.Sort(FileNumSorter);

        // remove out shape list
        fs.RemoveAt(0);
        return fs;
    }

    public IEnumerable<IGrouping<string, string>> GetFilesByGroup(string dir)
    {
        return GetFilesByOrdered(dir).GroupBy(file => string.Join(Separator, file.Split(Separator)[..2]));
    }

    /// <returns>dict:num$op_name -> num$op_name$param_name.</returns>.
    public Dictionary<string, IEnumerable<string>> GetFilesByOrderNum(string dir)
    {
        return GetFilesByGroup(dir)
            .ToDictionary(
            x =>
            {
                var split = x.Key.Split(Separator);
                return $"{split[0].Split(Path.DirectorySeparatorChar).Last()}{Separator}{split[1]}";
            },
            x => x.Select(s => s));
    }

    public OriginValue[] ExtractValues(string dir, Func<string, bool> extractor)
    {
        var fs = GetFilesByOrdered(dir);
        return fs
            .Where(filePath => extractor(Path.GetFileName(filePath)))
            .Select(path => new OriginValue(DataGenerator.FromTextFile(path), path))
            .ToArray();
    }

    public OriginValue[] GetComputeResults(string dir) => ExtractValues(dir, IsResultFile);

    public OriginValue GetComputeResult(string dir, int i)
    {
        var results = ExtractValues(dir, f => IsResultFile(f) && GetDumpFileNum(f) == i);
        Assert.NotEqual(results.Length, 0);
        return results.First();
    }

    public OriginValue[] GetParams(string dir, int count) => ExtractValues(
        dir,
        file => IsParamFile(file) && GetCount(file) == count);

    public OriginValue[] GetValues(string dir)
    {
        return ExtractValues(dir, _ => true);
    }

    public OriginValue[] OpExtract(string dir, string opName)
        => ExtractValues(dir, file => GetOpName(file) == opName);

    public OriginValue[] MatmulExtract(string dir)
    {
        return ExtractValues(dir, DynamicMatmulOnlyExtract).ToArray();
    }

    private bool FileFilter(string name)
    {
        return name != ".DS_Store" && !name.EndsWith("extcall");
    }
}
