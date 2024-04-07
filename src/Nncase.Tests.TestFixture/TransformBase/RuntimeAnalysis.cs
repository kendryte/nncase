// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Immutable;
using System.Linq;
using System.Text.RegularExpressions;
using DryIoc.ImTools;
using Nncase.IR;
using Nncase.Utilities;

namespace Nncase.Tests;

/// <summary>
/// dump output shape group by op.
/// </summary>
public static class RuntimeDumpAnalysis
{
    public static void ReadOutShapeList(string dumpResultRoot)
    {
        var data = MakeData(dumpResultRoot);
        PrintOutShapeList(data);
    }

    public static IEnumerable<IGrouping<string, (string Shape, int Index)>> MakeData(string dumpResultRoot)
    {
        using var stream = new StreamReader(Path.Join(dumpResultRoot, "!out_shape_list"));
        return GroupByOp(stream.ReadToEnd());
    }

    public static void PrintOutShapeList(IEnumerable<IGrouping<string, (string Shape, int Index)>> data)
    {
        foreach (var valueTuples in data)
        {
            Console.WriteLine($"op:{valueTuples.Key} count:{valueTuples.Count()}");
            foreach ((string x, int i) in valueTuples)
            {
                Console.WriteLine($"index:{i} shape:{x.Split(":")[1]}");
            }
        }
    }

    private static IEnumerable<IGrouping<string, (string Shape, int Index)>> GroupByOp(string str)
    {
        return str.Trim().Split("\n")
            .Select((x, i) => (x, i))
            .GroupBy(item => item.x.Split(":")[0]);
    }
}

/// <summary>
/// compare Result between Runtime and Evaluator
/// Runtime Data: read from dir/fileName
/// Evaluator Data: make call from Param Data in Runtime and run evaluate.
/// </summary>
public static class RuntimeResultAnalysis
{
    /// <param name="dir">dump data dir.</param>
    /// <param name="resultPath">resultPath for write cos.</param>
    /// <param name="ctor">call constructor.</param>
    public static void MatmulRun(string dir, string resultPath, Func<Expr[], Call> ctor)
    {
        var e = new TextDataExtractor();
        var data = e.MatmulExtract(dir);
        var cosList = data.Select(d => RuntimeResultAnalysis.Run(d.FileName, dir, ctor).First()).ToArray();
        using (var stream = File.OpenWrite(resultPath))
        {
            DumpUtility.WriteResult(stream, cosList);
        }
    }

    public static float[] Run(string fileName, string dir, Func<Expr[], Call> f)
    {
        // 1. get params
        var e = new TextDataExtractor();
        var number = DumpPathExtractor.GetCount(fileName);

        // todo: param sort
        var paramFiles = e.GetParams(dir, number);
        var lhs = paramFiles.FindFirst(x => x.FileName.Contains("lhs", StringComparison.Ordinal));
        var rhs = paramFiles.FindFirst(x => x.FileName.Contains("rhs", StringComparison.Ordinal));
        var param = new[] { lhs, rhs }.OrderBy(x => x.FileName.Last()).Select(x => x.Value).ToArray();

        var expect = e.GetComputeResult(dir, number);

        // 2. construct call
        var call = MakeCall(param, f);

        // 3. evaluate and run
        var result = call.Evaluate();
        var cos = Comparator.CosSimilarity(result, expect.Value);
        return cos;
    }

    private static Call MakeCall(IValue[] parameters, Func<Expr[], Call> f) =>
        f(parameters.Select(Const.FromValue).ToArray());
}

public static class ResultFinder
{
    public static OriginValue? FindFirstNanResult(string dir) =>
        FindFirst(
            dir,
            v => v.Value.AsTensor().ToArray<float>().Contains(float.NaN));

    public static OriginValue? FindFirstAll(string dir, Func<float, bool> fn) =>
        FindFirst(
            dir,
            v => v.Value.AsTensor().ToArray<float>().All(f => fn(f)));

    public static OriginValue? FindFirstAllZero(string dir) => FindFirstAll(dir, f => f == 0);

    public static OriginValue? FindFirstAllNaN(string dir) => FindFirstAll(dir, f => float.IsNaN(f));

    private static OriginValue? FindFirst(string dir, Func<OriginValue, bool> f) => new TextDataExtractor()
        .ExtractValues(dir, DumpPathExtractor.IsResultFile)
        .FindFirst(f);
}
