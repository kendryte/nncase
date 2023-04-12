// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.



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
