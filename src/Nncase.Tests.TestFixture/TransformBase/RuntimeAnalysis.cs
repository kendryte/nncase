// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Immutable;
using System.Text.RegularExpressions;
using DryIoc.ImTools;
using Nncase.IR;
using Nncase.Utilities;
using System.Linq;

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
        DumpUtility.WriteResult(resultPath, cosList);
    }

    public static float[] Run(string fileName, string dir, Func<Expr[], Call> f)
    {
        // 1. get params
        var e = new TextDataExtractor();
        var number = DumpPathExtractor.GetCount(fileName);
        // todo: param sort
        var paramFiles = e.GetParams(dir, number);
        var lhs = paramFiles.FindFirst(x => x.FileName.Contains("lhs"));
        var rhs = paramFiles.FindFirst(x => x.FileName.Contains("rhs"));
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

    // public static ValueRange<T> RangeOf<T>(Tensor t) where T : unmanaged, IEquatable<T>, IComparable<T> => QuantUtility.GetRange<T>();
    // todo: 寻找输入流，dump的文件和il关联，找到行为%num开始的，搜索括号
    // todo: prim func 获取不到参数，例如act
    // todo: evaluator和其他的地方evaluate dump到不同文件夹
    public static int OpIndexOf(string dir, int num)
    {
        var values = new TextDataExtractor()
            .ExtractValues(dir, DumpPathExtractor.IsResultFile);
        var targetOp = values[num];
        var opName = DumpPathExtractor.GetOpName(targetOp.FileName);
        return values.Where(v => DumpPathExtractor.GetOpName(v.FileName) == opName).ToArray().IndexOf(targetOp);
    }


    public class InputChainVisitor
    {
        private Action<Input> f1;
        private Action<Input[]> f2;
        private Dictionary<Input, Tensor> cache;

        public void Visit(Input root)
        {
            VisitImpl(root, new Input[]{});
        }

        private void VisitImpl(Input root, Input[] list)
        {
            if (root.Inputs.Count == 0)
            {
                f2(list);
                Console.WriteLine(string.Join(" ", list.Select(x => x.Index)));
                return;
            }

            foreach (var i in root.Inputs)
            {
                f1(i);
                VisitImpl(i, list.Concat(new[]{i}).ToArray());
            }
        }

        private void ValidText(Input[] inputs)
        {
            string dir = "";
            foreach (var input in inputs)
            {
                var values = new TextDataExtractor().GetValues(dir);
                // values[input.Index];
            }
        }
    }

    public static void InputChains()
    {
        string path = "/Users/homura/Code/nncase/tests_output/UnitTestK230Target/TestSplitEncWithMask/TargetDependentAfterQuantPass/9_InstStage/main.il";
        int num = 0;
        using var sr = new StreamReader(path);
        var inputs = sr.ReadToEnd().Split("\n").Select(x => x.Trim()).Where(x => Regex.IsMatch(x, @"^%\d")).ToArray();
        Console.WriteLine(string.Join("\n", inputs));
        var input = MakeInput(inputs, 63);
        new InputChainVisitor().Visit(input);
    }

    public static Input MakeInput(string[] lines, int index)
    {
        var expr = lines[index];
        var inputs = Regex.Match(expr, @"\(.*\)");
        if (inputs.Success)
        {
            // Console.WriteLine(inputs.Value);
            var operands = inputs.Value[1..^1].Split(",").Select(arg => ParseExprArg(lines, index, arg)).ToArray();
            var i = new Input(operands, index);
            return i;
        }

        throw new InvalidOperationException();
    }

    public class Input : IOperand
    {
        public Input(IOperand[] inputs, int index)
        {
            _operands = inputs;
            _index = index;
        }

        private IRArray<IOperand> _operands;

        public IRArray<IOperand> Operands => _operands;

        public IRArray<Input> Inputs => _operands.Where(operand => operand is Local).Select(x => ((Local)x).ToInput()).ToArray();

        private int _index;
        public int Index => _index;
    }

    public interface IOperand { }

    public record Arg(string name) : IOperand;

    public record Local(string[] lines, int id) : IOperand
    {
        public Input ToInput() => MakeInput(lines, id);
    }

    public record Constant(string value) : IOperand;

    public record OpEnum(string value) : IOperand;

    public static IOperand ParseExprArg(string[] lines, int index, string originArg)
    {
        // expr / const / op type
        var arg = originArg.Trim();
        if (arg.StartsWith("%"))
        {
            // var or local
            var value = arg[1..];
            if (int.TryParse(value, out var id))
            {
                return new Local(lines, id);
            }
            else
            {
                return new Arg(value);
            }
        }
        else if (arg.StartsWith("const"))
        {
            return new Constant(arg);
        }
        else
        {
            return new OpEnum(arg);
        }
    }
}

public static class ResultFinder
{
    private static OriginValue? FindFirst(string dir, Func<OriginValue, bool> f) => new TextDataExtractor()
        .ExtractValues(dir, DumpPathExtractor.IsResultFile)
        .FindFirst(f);

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
}
