﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
// [Theory]
//         [MemberData(nameof(DataOne))]
//         public void RunOne(ICodeGenCase Case) => RunCore(Case);

// protected void RunCore(ICodeGenCase Case)
//         {
//             var dumpDirPath = Path.Combine(DumpDirPath, Case.GetType().Name);
//             var opt = new RunPassOptions(null, 2, dumpDirPath);
//             // 1. get function

// var entry = Case.GetEntry();
//             var inferResult = entry.InferenceType();
//             entry.DumpAsScript("pre", dumpDirPath);
//             Assert.True(inferResult);

// // 2. run passes
//             var mod = new IRModule(entry);
//             var pmr = new PassManager(mod, opt);
//             pmr.Add(Case.GetPass());
//             pmr.Run();

// // 3. build re module and compare the function call
//             var rtmod = mod.ToRTModel(_target);
//             rtmod.Serialize();
//             rtmod.Dump("code", dumpDirPath);
//             Case.CompareEqual(rtmod);
//         }

// [Theory]
//         [MemberData(nameof(DataAll))]
//         public void RunAll(ICodeGenCase Case) => RunCore(Case);

// public static IEnumerable<object[]> DataOne => Data.Take(1);
//         public static IEnumerable<object[]> DataAll => Data.Skip(1);



// [Fact]
//         public void TestAdd()
//         {
//             var x = new Var("x", TensorType.Scalar(DataTypes.Float32));
//             var y = new Var("y", TensorType.Scalar(DataTypes.Float32));
//             var func = new Function(new Sequential() { x + y }, x, y);
//             var mod = new IRModule(func);
//             var rtmod = mod.ToRTModel(_target);
//             rtmod.Serialize();
//             Console.WriteLine(rtmod.Source);
//             Assert.Equal(3.5f, rtmod.Invoke(1.2f, 2.3f));
//         }

// [Fact]
//         public void TestCreateTarget()
//         {
//             var target = PluginLoader.CreateTarget("CSource");
//             Assert.Equal("CSource", target.Kind);
//         }
//     }
// }
