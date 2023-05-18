// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DryIoc.ImTools;
using Microsoft.Extensions.DependencyInjection;
using NetFabric.Hyperlinq;
using Nncase.CodeGen;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using Nncase.Passes.Transforms;
using Nncase.Tests.TestFixture;
using Nncase.Utilities;
using Xunit;
using Xunit.Abstractions;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.Tensors;
using Dimension = Nncase.IR.Dimension;
using Function = Nncase.IR.Function;
using Shape = Nncase.IR.Shape;

namespace Nncase.Tests.CoreTest;

[AutoSetupTestMethod(InitSession = true)]
public class UnitTestShapeSplitSegment : TestClassBase
{
    private readonly ITestOutputHelper _testOutputHelper;

    public UnitTestShapeSplitSegment(ITestOutputHelper testOutputHelper)
    {
        _testOutputHelper = testOutputHelper;
    }

    [Theory]
    [InlineData(64, 112)]
    [InlineData(112, 112)]
    [InlineData(190, 224)]
    [InlineData(224, 224)]
    public void TestSimpleShapeSplit(int dim, int expectDim)
    {
        var inVar = new Var(new TensorType(DataTypes.Float32, new[] { 1, 3, Dimension.Unknown, 224 }));
        var f = new Function((inVar * 2f) + 1f, inVar);
        var info = new[] { new SegmentInfo(0, 2, new[] { 112, 224 }) };
        var module = new ShapeBucket().Run(f, info, CompileSession.CompileOptions);
        var (_, kmodel) = Testing.BuildKModel("TestSimple.kmodel", module, CompileSession);
        var input = new[] { Testing.Rand<float>(1, 3, dim, 224) };
        var actual = Testing.RunKModel(kmodel, Dumpper.Directory, input).AsTensors()[0];
        Assert.Equal(expectDim, actual.Shape[2]);
        var expect =
            f.Body.Evaluate(new Dictionary<Var, IValue> { { inVar, Value.FromTensor(input[0]) } }).AsTensors()[0];
        if (dim == expectDim)
        {
            Assert.True(Comparator.CosSimilarity(expect, actual) > 0.99);
        }
    }

    // [Fact]
    // public void TestOutOfSegment()
    // {
    //     var inVar = new Var(new TensorType(DataTypes.Float32, new[] { 1, 3, Dimension.Unknown, 224 }));
    //     var f = new Function(inVar, new[] { inVar });
    //     var info = new SegmentInfo(0, 2, new[] { 112, 224 });
    //     var module = new ShapeSplitSegment().Run(f, info);
    //     var (_, kmodel) = Testing.BuildKModel("TestSimple.kmodel", module, CompileSession);
    //     var input = new[] { Testing.Rand<float>(1, 3, 225, 224) };
    //     Assert.Throws<InvalidOperationException>(
    //         () => Testing.RunKModel(kmodel, Dumpper.Directory, input));
    [Fact]
    public void TestOutOfSegment()
    {
        var inVar = new Var(new TensorType(DataTypes.Float32, new[] { 1, 3, Dimension.Unknown, 224 }));
        var f = new Function(inVar, new[] { inVar });
        var info = new[] { new SegmentInfo(0, 2, new[] { 112, 224 }) };
        var module = new ShapeBucket().Run(f, info, CompileSession.CompileOptions);
        var (_, kmodel) = Testing.BuildKModel("TestSimple.kmodel", module, CompileSession);
        var input = new[] { Testing.Rand<float>(1, 3, 225, 224) };
        Assert.Throws<InvalidOperationException>(
            () => Testing.RunKModel(kmodel, Dumpper.Directory, input));
    }

    [Fact]
    public void TestMultiVar()
    {
        // todo: a method generate var and random data
        var shape = new[] { 1, 3, 2, 1 };
        var inVar0 = new Var(new TensorType(DataTypes.Float32, shape));
        var f = new Function(new If(false,
                new If(true, IR.F.Tensors.ShapeOf(inVar0) + 0L, IR.F.Tensors.ShapeOf(inVar0) + 1L),
                new If(true, inVar0, IR.F.Tensors.ShapeOf(inVar0) + 3L)),
            inVar0);

        var input = DataGenerator.DefaultRandom(shape).Evaluate().AsTensors();
        var module = new IRModule(f);
        CompilerServices.DumpIR(f, "if_multi_var",
            "/Users/homura/Code/nncase/tests_output/UnitTestShapeSplitSegment/TestMultiInputShapeSplit/");
        var (kmodelPath, kmodel) = Testing.BuildKModel("TestSimple.kmodel", module, CompileSession);
        Testing.RunKModel(kmodel, Dumpper.Directory, input);
    }

    [Fact]
    public void TestMultiInputAndVar()
    {
        var shape = new[] { 1, 3, 2, 1 };
        var inVar0 = new Var(new TensorType(DataTypes.Float32, shape));
        var inVar1 = new Var(new TensorType(DataTypes.Float32, shape));
        var f = new Function(new If(false,
                new If(true, IR.F.Tensors.ShapeOf(inVar0) + IR.F.Tensors.ShapeOf(inVar1),
                    IR.F.Tensors.ShapeOf(inVar0) + IR.F.Tensors.ShapeOf(inVar1) + 1L),
                new If(true, inVar0 + inVar1, inVar0 + inVar1 + 1)),
            new[] { inVar0, inVar1 });

        var input = DataGenerator.DefaultRandom(shape).Evaluate().AsTensors()[0];
        var module = new IRModule(f);
        CompilerServices.DumpIR(f, "if_multi_var",
            "/Users/homura/Code/nncase/tests_output/UnitTestShapeSplitSegment/TestMultiInputShapeSplit/");
        var (kmodelPath, kmodel) = Testing.BuildKModel("TestSimple.kmodel", module, CompileSession);
        Testing.DumpInterpModel(kmodelPath, new[] { input, input }, Dumpper.Directory);
        Testing.RunKModel(kmodel, Dumpper.Directory, new[] { input, input });
    }

    [Fact]
    public void TestMultiInput()
    {
        SegmentInfo[] infos = { new(0, 2, new[] { 16, 32 }), new(1, 2, new[] { 24, 48 }), };
        var inVar0 = new Var(new TensorType(DataTypes.Float32, new[] { 1, 3, Dimension.Unknown, 224 }));
        var inVar1 = new Var(new TensorType(DataTypes.Float32, new[] { 1, 3, Dimension.Unknown, 128 }));
        var f = new Function((inVar0 * 2f) + inVar1, inVar0, inVar1);
        var result = new ShapeBucket().Run(f, infos, CompileOptions);
        Dumpper.DumpIR(result.Entry!, "multi_input_if");
    }

    [Fact]
    public void ResultAnalysis()
    {
        var dir = Path.Join("/Users/homura/Code/nncase/tests_output/UnitTestShapeSplitSegment/TestSplitEncWithMask/",
            "Result");
        var resultPath = Path.Join(dir, "result");
        var expectPath = Path.Join(dir, "expect");
        var result = DataGenerator.FromTextFile(resultPath);
        var expect = DataGenerator.FromTextFile(expectPath);
        DetailComparator.GenerateFullCompareInfo(
            new[] { new OriginValue(result, resultPath) },
            new[] { new OriginValue(expect, expectPath) },
            Path.Join(dir, "Analysis"));
    }

    [Fact]
    public void FindFirstNaN()
    {
        var runtimeDir = "/Users/homura/Code/nncase/tests_output/UnitTestK230Target/TestSplitEncWithMask/Runtime/";
        var firstNan = ResultFinder.FindFirstAllNaN(runtimeDir);

        if (firstNan != null)
        {
            _testOutputHelper.WriteLine(firstNan.FileName);
            var n = DumpPathExtractor.GetCount(firstNan.FileName);
            var op = DumpPathExtractor.GetOpName(firstNan.FileName);
            var i = RuntimeResultAnalysis.OpIndexOf(runtimeDir, n);
            _testOutputHelper.WriteLine($"{op} {i} is all nan");
        }
    }

    [Fact]
    public void EncMatmulAnalysis()
    {
        // todo: find input chains for some result
        var e = new TextDataExtractor();
        var runtimeDir =
            "/Users/homura/Code/nncase/tests_output/UnitTestShapeSplitSegment/TestSplitEncWithMask/Runtime";
        var evaluatorDir =
            "/Users/homura/Code/nncase/tests_output/UnitTestShapeSplitSegment/TestSplitEncWithMask/Evaluator";
        var runtimeData = e.MatmulExtract(runtimeDir);
        var evaluatorData = e.MatmulExtract(runtimeDir);
        var firstNan = e.ExtractValues(runtimeDir, DumpPathExtractor.IsResultFile)
            .FindFirst(v => v.Value.AsTensor().ToArray<float>().Contains(float.NaN));
        // _testOutputHelper.WriteLine(firstNan.Path);
        int i = 0;
        foreach (var (item1, item2) in runtimeData.Zip(evaluatorData))
        {
            var dir = "/Users/homura/Code/nncase/tests_output/UnitTestShapeSplitSegment/TestSplitEncWithMask/Compare/";
            Directory.CreateDirectory(dir);
            var iStr = i.ToString().PadLeft(runtimeData.Length.ToString().Length, '0');
            DumpUtility.WriteResult(Path.Join(dir, $"{iStr}_runtime"), item1.AsTensor().Tensor.ToArray<float>());
            DumpUtility.WriteResult(Path.Join(dir, $"{iStr}_eval"), item2.AsTensor().Tensor.ToArray<float>());
            i++;
        }

        RuntimeResultAnalysis.MatmulRun(
            runtimeDir,
            "/Users/homura/Code/nncase/tests_output/UnitTestShapeSplitSegment/TestSplitEncWithMask/call_runtime/AnalysisResult.log",
            args => IR.F.Math.MatMul(args.ToArray()[0], args.ToArray()[1]));
    }

    public void DumpInputs(Tensor[] inputs)
    {
        var path = "/Users/homura/Code/nncase/tests_output/UnitTestShapeSplitSegment/TestSplitEncWithMask/Inputs/";

        Directory.CreateDirectory(path);
        for (int i = 0; i < inputs.Length; i++)
        {
            DumpUtility.WriteResult(Path.Join(path, $"{i}.txt"), inputs[i].GetArrayString());
        }
    }


    private static Expr FitFixedShape(Expr targetInput, int[] fixedShape)
    {
        targetInput.InferenceType();
        // rename
        // compute paddings;
        var pads = fixedShape - Cast(ShapeOf(targetInput), DataTypes.Int32);
        var paddings = Transpose(Stack(new IR.Tuple(Enumerable.Repeat(0, fixedShape.Length).ToArray(), pads), 0),
            new[] { 1, 0 });
        var fixedInput = IR.F.NN.Pad(targetInput, paddings, PadMode.Constant,
            Cast(0f, targetInput.CheckedDataType));
        return fixedInput;
    }
    // todo: codegen的时候dump每个expr对应的pc

    [Fact]
    public async Task TestSplitEncWithMask()
    {
        // todo:找到第一个nan层
        CompileOptions.DumpFlags = DumpFlags.Rewrite | DumpFlags.Evaluator | DumpFlags.PassIR;
        // var path = "/Users/homura/tmp/22_11_Model/z2e/enc_zho.onnx";
        var path = "/Users/homura/tmp/fix_shape/tts_enc_zho.onnx";
        var info = new[] { new SegmentInfo(0, 1, new[] { 64, 128 }), new SegmentInfo(1, 1, new[] { 64, 128 }), };
        var dim = 72;
        var expectDim = 128;


        // todo: int type random

        // padding in left
        // var text = Testing.Rand<long>(1, dim);
        var text = Tensor.From(
            new long[]
            {
                25, 26, 7, 4, 1, 10, 23, 37, 7, 1, 39, 20, 10, 1, 43, 11, 24, 5, 25, 26, 7, 4, 1, 10, 23, 37, 7, 1,
                39, 20, 10, 1, 43, 11, 24, 5, 25, 26, 7, 4, 1, 10, 23, 37, 7, 1, 39, 20, 10, 1, 43, 11, 24, 5, 25,
                26, 7, 4, 1, 10, 23, 37, 7, 1, 39, 20, 10, 1, 43, 11, 24, 5, 25, 26, 7, 4, 1, 10, 23, 37, 7, 1, 39,
                20, 10, 1, 43, 11, 24, 5, 25, 26, 7, 4, 1, 10, 23, 37, 7, 1, 39, 20, 10, 1, 43, 11, 24, 5, 25, 26,
                7, 4, 1, 10, 23, 37, 7, 1, 39, 20, 10, 1, 43, 11, 24, 5, 25, 26, 7, 4, 1, 10, 23, 37, 7, 1, 39, 20,
                10, 1, 43, 11, 24, 5, 25, 26, 7, 4, 1, 10, 23, 37, 7, 1, 39, 20, 10, 1, 43, 11, 24, 5, 25, 26, 7, 4,
                1, 10, 23, 37, 7, 1, 39, 20, 10, 1, 43, 11, 24, 5, 25, 26, 7, 4, 1, 10, 23, 37, 7, 1, 39, 20, 10, 1,
                43, 11, 24, 5, 25, 26, 7, 4, 1, 10, 23, 37, 7, 1, 39, 20, 10, 1, 43, 11, 24, 5, 25, 26, 7, 4, 1, 10,
                23, 37, 7, 1, 39, 20, 10, 1, 43, 11, 24, 5, 25, 26, 7, 4, 1, 10, 23, 37, 7, 1, 39, 20, 10, 1, 43,
                11, 24, 5, 35, 24, 15, 5,
            }.Take(dim).ToArray(), new[] { 1, dim });
        // var text = FitFixedShape(originText, new[] { 1, expectDim }).Evaluate().AsTensor();

        // var trueMask = Enumerable.Repeat(true, expectDim - dim);
        var mask = Tensor.From(Enumerable.Repeat(false, dim).ToArray(), new[] { 1, dim });
        // var mask = Tensor.From(
            // falseMask.Concat(trueMask).ToArray(),
            // Enumerable.Repeat(true, dim).ToArray(),
            // new[] { 1, expectDim });

        // var spk = Testing.Rand<long>(1);
        // var spk = Abs(IR.F.Random.Normal(DataType.FromType<long>(), 0, 10, 1, new[] { 1 })).Evaluate().AsTensor()
        //     .Cast<long>();

        // var spd = Testing.Rand<float>(1);
        var spk = Tensor.From(new long[] { 0 });
        var spd = Tensor.From(new[] { 1f });
        var input = new Tensor[] { text, mask, spk, spd };
        DumpInputs(input);


        // import
        using var file = File.OpenRead(path);
        var tmpModule = Importers.ImportOnnx(file, CompileSession);
        var m = PreProcess(tmpModule);
        // Dumpper.DumpModule(m, "module");

        // split
        var module = new ShapeBucket().Run((Function)m.Entry!, info, CompileOptions);
        await InferShapeAsync(module);
        Dumpper.DumpModule(module, "module");

        // run
        var (kmodelPath, kmodel) = Testing.BuildKModel("TestSimple", module, CompileSession);
        Testing.DumpInterpModel(kmodelPath, input, Path.Join(Dumpper.Directory, "call_runtime"));
        var actual = Testing.RunKModel(kmodel, Dumpper.Directory, input).AsTensors();
        // var result = PostProcess(actual, mask);

        Directory.CreateDirectory(Path.Join(Dumpper.Directory, "Result"));
        ValueDumper.DumpTensor(actual[0], Path.Join(Dumpper.Directory, "Result", "actual__0"));
        ValueDumper.DumpTensor(actual[1], Path.Join(Dumpper.Directory, "Result", "actual__1"));

        // // todo: 不能在任意位置重新修改Flag
        // // evaluate
        // await InferShapeAsync(m);
        // var f = (Function)m.Entry!;
        // // var expect = f.Body.Evaluate(new Dictionary<Var, IValue>
        // //     {
        // //         { f.Parameters[0], Value.FromTensor(originText) },
        // //         { f.Parameters[1], Value.FromTensor(Tensor.From(Enumerable.Repeat(false, dim).ToArray(), new[]{1, dim})) },
        // //         { f.Parameters[2], Value.FromTensor(input[2]) },
        // //         { f.Parameters[3], Value.FromTensor(input[3]) },
        // //     })
        // //     .AsTensors()[0];
        // var (_, expectKmodel) = Testing.BuildKModel("TestSimple", tmpModule, CompileSession);
        // var expect = Testing.RunKModel(expectKmodel, Dumpper.Directory, input).AsTensors();
        //
        // ValueDumper.DumpTensor(expect[0], Path.Join(Dumpper.Directory, "Result", "expect__0"));
        // ValueDumper.DumpTensor(expect[1], Path.Join(Dumpper.Directory, "Result", "expect__1"));
        // // DumpUtility.WriteResult(Path.Join(Dumpper.Directory, "Result", "expect"), expect.ToArray<float>(),
        // //     DumpUtility.SerializeShape(expect.Shape) + "\n");
        //
        // // compare
        // var cos0 = Comparator.CosSimilarity(expect[0], actual[0]);
        // var cos1 = Comparator.CosSimilarity(expect[1], actual[1]);
        // _testOutputHelper.WriteLine($"cos0:{cos0}");
        // _testOutputHelper.WriteLine($"cos1:{cos1}");
        // Assert.True(cos0 > 0.99);
        // Assert.True(cos1 > 0.99);
    }

    [Fact]
    public async Task TestSplitEnc()
    {
        CompileOptions.DumpFlags = DumpFlags.Rewrite;
        var path = "/Users/homura/tmp/22_11_Model/z2e/enc_zho.onnx";
        var info = new[] { new SegmentInfo(0, 1, new[] { 64, 128 }) };
        var dim = 128;
        var expectDim = 128;
        var input = new[] { Testing.Rand<float>(1, dim, 192) };
        await TestNMTModel(path, info, input, dim, expectDim);
    }

    private Task InferShapeAsync(IRModule module)
    {
        var pmgr = CompileSession.CreatePassManager("InferShapePasses");
        pmgr.Add<ShapeInferPass>();
        return pmgr.RunAsync(module);
    }

    [Fact]
    public async Task TestSplitEncOrigin()
    {
        using var file = File.OpenRead("/Users/homura/tmp/22_11_Model/z2e/enc_zho.onnx");
        var module = Importers.ImportOnnx(file, CompileSession);
        await InferShapeAsync(module);
        Dumpper.DumpModule(module, "module");

        // var inVar = new Var(new TensorType(DataTypes.Float32, new[] { 1, 3, Dimension.Unknown, 224 }));
        // var f = new Function((inVar * 2f) + 1f, inVar);
        // var info = new SegmentInfo(0, 2, new[] { 112, 224 });
        // var module = new ShapeSplitSegment().Run(f, info);

        var (kmodelPath, kmodel) = Testing.BuildKModel("TestSimple", module, CompileSession);

        var dim = 128;
        var expectDim = 128;
        var input = new[] { Testing.Rand<float>(1, dim, 192) };
        Testing.DumpInterpModel(kmodelPath, input, Path.Join(Dumpper.Directory, "call_runtime"));
        var actual = Testing.RunKModel(kmodel, Dumpper.Directory, input).AsTensors()[0];
        var f = (Function)module.Entry!;
        Assert.Equal(expectDim, actual.Shape[2]);
        var expect = f.Body.Evaluate(new Dictionary<Var, IValue> { { f.Parameters[0], Value.FromTensor(input[0]) } })
            .AsTensors()[0];
        if (dim == expectDim)
        {
            Assert.True(Comparator.CosSimilarity(expect, actual) > 0.99);
        }
    }

    [Fact]
    public void TestSplitDecFirst()
    {
        var path = "/Users/homura/tmp/22_11_Model/z2e/dec_first.onnx";
        var info = new[] { new SegmentInfo(0, 1, new[] { 64, 128 }), new SegmentInfo(1, 1, new[] { 64, 128 }) };
        var dim = 128;
        var expectDim = 128;
        var input = new[] { Testing.Rand<long>(1, dim), Testing.Rand<long>(1, dim) };
        TestNMTModel(path, info, input, dim, expectDim);
    }

    [Fact]
    public void TestSplitDecNext()
    {
        var path = "/Users/homura/tmp/22_11_Model/z2e/dec_next.onnx";
        var info = new[] { new SegmentInfo(0, 1, new[] { 64, 128 }), new SegmentInfo(1, 1, new[] { 64, 128 }) };
        var dim = 128;
        var expectDim = 128;
        var input = new[] { Testing.Rand<long>(1, dim), Testing.Rand<long>(1, dim) };
        TestNMTModel(path, info, input, dim, expectDim);
    }

    private IRModule PreProcess(IRModule m)
    {
        var originF = (Function)m.Entry!;
        var newInput = new Var("dynamic_input", new TensorType(DataTypes.Int64, new[] { 1, Dimension.Unknown }));
        var newMask = new Var("dynamic_mask", new TensorType(DataTypes.Boolean, new[] { 1, Dimension.Unknown }));

        var tmpBody = ReplaceUtility.ReplaceExpr(originF.Body, originF.Parameters[0], newInput);
        var newBody = ReplaceUtility.ReplaceExpr(tmpBody, originF.Parameters[1], newMask);
        var f = new Function("main_after_process", newBody,
            new[] { newInput, newMask, originF.Parameters[2], originF.Parameters[3] });
        return new IRModule(f);
    }

    public Tensor PostProcess(Tensor encOut, Tensor mask)
    {
        var maskValue = mask.ToArray<bool>();
        var i = maskValue.IndexOf(x => !x);
        var output = Slice(encOut, new[] { 0, i, 0 }, encOut.Shape.ToValueArray(), 3).Evaluate().AsTensor();
        return output;
    }

    private async Task TestNMTModel(string path, SegmentInfo[] info, Tensor[] input, int dim, int expectDim)
    {
        using var file = File.OpenRead(path);
        var tmpModule = Importers.ImportOnnx(file, CompileSession);
        var m = PreProcess(tmpModule);
        var module = new ShapeBucket().Run((Function)m.Entry!, info, CompileOptions);
        module.Entry!.InferenceType();
        await InferShapeAsync(module);

        Dumpper.DumpModule(module, "module");
        var (kmodelPath, kmodel) = Testing.BuildKModel("TestSimple", module, CompileSession);
        Testing.DumpInterpModel(kmodelPath, input, Path.Join(Dumpper.Directory, "call_runtime"));
        var actual = Testing.RunKModel(kmodel, Dumpper.Directory, input).AsTensors()[0];
        var f = (Function)module.Entry!;
        // todo: fix
        // Assert.Equal(expectDim, actual.Shape[1]);
        var expect = f.Body.Evaluate(new Dictionary<Var, IValue> { { f.Parameters[0], Value.FromTensor(input[0]) } })
            .AsTensors()[0];
        if (dim == expectDim)
        {
            Assert.True(Comparator.CosSimilarity(expect, actual) > 0.99);
        }
    }

    [Fact]
    public void TestLocalAlloc()
    {
        var a = new Var(new TensorType(DataTypes.Boolean, Shape.Scalar));
        var b = new Var(new TensorType(DataTypes.Int32, Shape.Scalar));
        var c = new Var(new TensorType(DataTypes.Int32, Shape.Scalar));
        var va = (Tensor)true;
        var vb = 1;
        var vc = 2;
        var inputs = new[] { va, vb, vc };
        var body = new If(a, b + c, b + c + 1);
        var f = new Function(body, a, b, c);
        var m = new IRModule(f);
        Assert.True(m.Entry != null && m.Entry.InferenceType());
        var (path, kmodel) = Testing.BuildKModel("TestLocalAlloc", m, CompileSession);
        var dumpDir = Dumpper.Directory;
        Testing.DumpInterpModel(path, inputs, dumpDir);
        var result = Testing.RunKModel(kmodel, dumpDir, inputs).AsTensor();
        var expect = f.Body.Evaluate(new Dictionary<Var, IValue>
        {
            { a, Value.FromTensor(va) }, { b, Value.FromTensor(vb) }, { c, Value.FromTensor(vc) },
        }).AsTensor();
        Assert.Equal(result, expect);
    }




    // public static void TestTimer()
    // {
    //     using (var _ = new Timer("Test"))
    //     {
    //         Console.WriteLine("write 1");
    //         Console.WriteLine("write 2");
    //     }
    //
    //     Console.WriteLine("write 3");
    // }
}
