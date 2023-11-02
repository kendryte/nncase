// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.Passes;
using Nncase.Passes.Mutators;
using Nncase.Tests.TestFixture;
using Nncase.TIR;
using Xunit;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.Tensors;

namespace Nncase.Tests.TIRTest;

[AutoSetupTestMethod(InitSession = true)]
public sealed class UnitTestMutators : TestClassBase
{
    public UnitTestMutators()
    {
#if DEBUG
        CompileOptions.DumpFlags = Diagnostics.DumpFlags.Rewrite;
#endif
    }

    [Fact]
    public async Task TestFoldConstCallWithTuple()
    {
        T.CreateBuffer(new TensorType(DataTypes.BFloat16, new[] { 48 }), MemoryLocation.Input, out var ddr_if);
        T.CreateBuffer(new TensorType(DataTypes.BFloat16, new[] { 9 }), MemoryLocation.Data, out var glb_if_ping);
        T.CreateBuffer(new TensorType(DataTypes.BFloat16, new[] { 9 }), MemoryLocation.Data, out var glb_if_pong);
        PrimFunction main;
        {
            main = T.PrimFunc("main", Callable.StackVMModuleKind, ddr_if).Body(
                   T.Unrolled(out var w, (0, 48, 9)).Body(
                     new Call(
                         new LoadT(),
                         new BufferRegion(
                           ddr_if,
                           new Range[] { (w, Min(w + 9, 48)) }),
                         GetItem(
                             new Tuple(new[] {
                          new BufferRegion(
                              glb_if_ping,
                              new Range[] { (0, Min(w + 9, 48) - w) }),
                          new BufferRegion(
                              glb_if_pong,
                              new Range[] { (0, Min(w + 9, 48) - w) }),
                          }),
                             Mod(w / 9, 2)))))
           .Build();
        }

        var prmg = CompileSession.CreatePassManager("prmg");
        prmg.AddWithName<PrimFuncPass>("unroll").Configure(
          p =>
          {
              p.Add<UnRollLoopSequential>();
              p.Add<FoldConstCall>();
          });

        var module = new IR.IRModule(main);
        await prmg.RunAsync(module);

        var post = (TIR.PrimFunction)module.Entry!;
        {
            var getBuffer = (int i, ParameterInfo info) =>
            {
                var bufferRegion = (BufferRegion)((Call)post.Body.Fields[i])[info];
                return bufferRegion.Buffer;
            };
            int count = 0;
            for (int w = 0; w < 48; w += 9)
            {
                // Assert.True(object.ReferenceEquals(getBuffer(count, LoadT.DdrPp), post.Parameters[0]));
                var name = getBuffer(count++, LoadT.GlbPp).Name[^4..];

                // System.Console.WriteLine($"{w} {name}");
                if (((w / 9) % 2) == 0)
                {
                    Assert.Equal("ping", name);
                }
                else
                {
                    Assert.Equal("pong", name);
                }
            }
        }
    }

    [Fact]
    public async Task TestUnRollLoopSequential()
    {
        var main = T.PrimFunc("main", Callable.StackVMModuleKind).Body(// (*i8) -> ()
            T.Unrolled(out var i, (0, 32, 4)).Body(// ()
              T.Unrolled(out var j, (0, 16, 4)).Body(// ()
                T.Unrolled(out var k, (0, 18, 6)).Body(// ()
                  T.Unrolled(out var l, (0, 32, 16)).Body(// ()
                    T.Block("block").Body()))))).Build();

        CompilerServices.InferenceType(main);
        var prmg = CompileSession.CreatePassManager("prmg");
        prmg.AddWithName<PrimFuncPass>("unroll").Configure(
          p =>
          {
              p.Add<UnRollLoopSequential>();
          });

        var module = new IR.IRModule(main);
        await prmg.RunAsync(module);
        Assert.Equal(32 / 4 * (16 / 4) * (18 / 6) * (32 / 16), ((TIR.PrimFunction)module.Entry!).Body.Count);
    }

    [Fact]
    public async Task TestUnRollLoopSequential2()
    {
        T.CreateBuffer(new TensorType(DataTypes.BFloat16, new[] { 3, 16, 24, 24 }), MemoryLocation.Input, out var ddr_if);
        T.CreateBuffer(new TensorType(DataTypes.BFloat16, new[] { 3, 10, 5, 9 }), MemoryLocation.Data, out var glb_if);

        PrimFunction main;
        {
            main = T.PrimFunc("main", Callable.StackVMModuleKind).Body(
             T.Unrolled(out var n, (0, 3, 3)).Body(
               T.Unrolled(out var c, (0, 16, 10)).Body(
                 T.Unrolled(out var h, (0, 24, 5)).Body(
                   T.Unrolled(out var w, (0, 24, 9)).Body(
                     new Call(
                         new LoadT(),
                         new BufferRegion(
                           ddr_if,
                           new TIR.Range[] { (n, IR.F.Math.Min(n + 3, 3)),
                                        (c, IR.F.Math.Min(c + 10, 16)),
                                        (h, IR.F.Math.Min(h + 5, 24)),
                                        (w, IR.F.Math.Min(w + 9, 24)),
                                         }),
                         new BufferRegion(
                           glb_if,
                           new TIR.Range[] { (0, IR.F.Math.Min(n + 3, 3) - n),
                                        (0, IR.F.Math.Min(c + 10, 16) - c),
                                        (0, IR.F.Math.Min(h + 5, 24) - h),
                                        (0, IR.F.Math.Min(w + 9, 24) - w),
                                         })))))))
           .Build();
        }

        CompilerServices.InferenceType(main);
        var prmg = CompileSession.CreatePassManager("prmg");
        prmg.AddWithName<PrimFuncPass>("unroll").Configure(
          p =>
          {
              p.Add<UnRollLoopSequential>();
          });

        var module = new IR.IRModule(main);
        await prmg.RunAsync(module);
        var post = (TIR.PrimFunction)module.Entry!;
        {
            var getRegion = (int i, ParameterInfo info) =>
            {
                var bufferRegion = (BufferRegion)((Call)post.Body.Fields[i])[info];
                return bufferRegion.Region.AsValueEnumerable().Select(rg =>
                  rg.Start.Evaluate().AsTensor().ToScalar<int>()..rg.Stop.Evaluate().AsTensor().ToScalar<int>())
                  .ToArray();
            };
            int count = 0;
            for (int n = 0; n < 3; n += 3)
            {
                for (int c = 0; c < 16; c += 10)
                {
                    for (int h = 0; h < 24; h += 5)
                    {
                        for (int w = 0; w < 24; w += 9)
                        {
                            var ddrPp = new System.Range[] {
                                        n..System.Math.Min(n + 3, 3),
                                        c..System.Math.Min(c + 10, 16),
                                        h..System.Math.Min(h + 5, 24),
                                        w..System.Math.Min(w + 9, 24),
                            };
                            Assert.True(ddrPp.SequenceEqual(getRegion(count, LoadT.DdrPp)));
                            var glbPp = new System.Range[] {
                                        0..(System.Math.Min(n + 3, 3) - n),
                                        0..(System.Math.Min(c + 10, 16) - c),
                                        0..(System.Math.Min(h + 5, 24) - h),
                                        0..(System.Math.Min(w + 9, 24) - w),
                            };

                            Assert.True(glbPp.SequenceEqual(getRegion(count, LoadT.GlbPp)));
                            count++;
                        }
                    }
                }
            }
        }
    }

    [Fact]
    public async Task TestUnRollLoopSequential3()
    {
        T.CreateBuffer(new TensorType(DataTypes.BFloat16, new[] { 3, 16, 24, 24 }), MemoryLocation.Input, out var ddr_if);
        T.CreateBuffer(new TensorType(DataTypes.BFloat16, new[] { 3, 10, 5, 9 }), MemoryLocation.Data, out var glb_if);

        PrimFunction main;
        {
            main = T.PrimFunc("main", Callable.StackVMModuleKind).Body(
             T.Unrolled(out var n, (0, 3, 3)).Body(
               T.Unrolled(out var c, (0, 16, 10)).Body(
                 T.Unrolled(out var h, (0, 24, 5)).Body(
                   T.Unrolled(out var w, (0, 24, 9)).Body(
                     new Call(
                         new LoadT(),
                         new BufferRegion(
                           ddr_if,
                           new TIR.Range[] { (n, IR.F.Math.Min(n + 3, 3)),
                                        (c, IR.F.Math.Min(c + 10, 16)),
                                        (h, IR.F.Math.Min(h + 5, 24)),
                                        (w, IR.F.Math.Min(w + 9, 24)),
                                         }),
                         new BufferRegion(
                           glb_if,
                           new TIR.Range[] { (0, IR.F.Math.Min(n + 3, 3) - n),
                                        (0, IR.F.Math.Min(c + 10, 16) - c),
                                        (0, IR.F.Math.Min(h + 5, 24) - h),
                                        (0, IR.F.Math.Min(w + 9, 24) - w),
                                         })),
                     T.Unrolled(out var tcu_h, (h, IR.F.Math.Min(h + 5, 24), 2)).Body(
                      new Call(
                         new LoadT(),
                         new BufferRegion(
                           ddr_if,
                           new TIR.Range[] { (n, IR.F.Math.Min(n + 3, 3)),
                                        (c, IR.F.Math.Min(c + 10, 16)),
                                        (h + tcu_h, IR.F.Math.Min(h + tcu_h + 2, 24)),
                                        (w, IR.F.Math.Min(w + 9, 24)),
                                         }),
                         new BufferRegion(
                           glb_if,
                           new TIR.Range[] { (0, IR.F.Math.Min(n + 3, 3) - n),
                                        (0, IR.F.Math.Min(c + 10, 16) - c),
                                        (tcu_h, IR.F.Math.Min(h + tcu_h + 2, 24) - h),
                                        (0, IR.F.Math.Min(w + 9, 24) - w),
                                         }))))))))
           .Build();
        }

        CompilerServices.InferenceType(main);
        var prmg = CompileSession.CreatePassManager("prmg");
        prmg.AddWithName<PrimFuncPass>("unroll").Configure(
          p =>
          {
              p.Add<UnRollLoopSequential>();
          });

        var module = new IR.IRModule(main);
        await prmg.RunAsync(module);
        var post = (TIR.PrimFunction)module.Entry!;
        {
            var getRegion = (int i, ParameterInfo info) =>
            {
                var bufferRegion = (BufferRegion)((Call)post.Body.Fields[i])[info];
                return bufferRegion.Region.AsValueEnumerable().Select(rg =>
                  rg.Start.Evaluate().AsTensor().ToScalar<int>()..rg.Stop.Evaluate().AsTensor().ToScalar<int>())
                  .ToArray();
            };
            int count = 0;
            for (int n = 0; n < 3; n += 3)
            {
                for (int c = 0; c < 16; c += 10)
                {
                    for (int h = 0; h < 24; h += 5)
                    {
                        for (int w = 0; w < 24; w += 9)
                        {
                            var ddrPp = new System.Range[] {
                                        n..System.Math.Min(n + 3, 3),
                                        c..System.Math.Min(c + 10, 16),
                                        h..System.Math.Min(h + 5, 24),
                                        w..System.Math.Min(w + 9, 24),
                            };
                            Assert.True(ddrPp.SequenceEqual(getRegion(count, LoadT.DdrPp)));
                            var glbPp = new System.Range[] {
                                        0..(System.Math.Min(n + 3, 3) - n),
                                        0..(System.Math.Min(c + 10, 16) - c),
                                        0..(System.Math.Min(h + 5, 24) - h),
                                        0..(System.Math.Min(w + 9, 24) - w),
                            };
                            Assert.True(glbPp.SequenceEqual(getRegion(count, LoadT.GlbPp)));
                            count++;
                            for (int tcu_h = h; tcu_h < System.Math.Min(h + 5, 24); tcu_h += 2)
                            {
                                var ddrPp2 = new System.Range[] {
                                        n..System.Math.Min(n + 3, 3),
                                        c..System.Math.Min(c + 10, 16),
                                        (h + tcu_h)..System.Math.Min(h + tcu_h + 2, 24),
                                        w..System.Math.Min(w + 9, 24),
                                };
                                Assert.True(ddrPp2.SequenceEqual(getRegion(count, LoadT.DdrPp)));
                                var glbPp2 = new System.Range[] {
                                        0..(System.Math.Min(n + 3, 3) - n),
                                        0..(System.Math.Min(c + 10, 16) - c),
                                        tcu_h..(System.Math.Min(h + tcu_h + 2, 24) - h),
                                        0..(System.Math.Min(w + 9, 24) - w),
                                };
                                Assert.True(glbPp2.SequenceEqual(getRegion(count, LoadT.GlbPp)));
                                count++;
                            }
                        }
                    }
                }
            }
        }
    }

    [Fact]
    public async Task TestFoldLet()
    {
        var main = T.PrimFunc("main", Callable.StackVMModuleKind).Body(// (*i8) -> ()
          T.Unrolled(out var i, (0, 32, 4)).Body(// ()
            T.Let(out var a, (Expr)10 - (Expr)2).Body(
              T.Let(out var b, (Expr)10 + (Expr)2).Body(
                new Call(new ExtraW(), i + a + b)))))
        .Build();

        CompilerServices.InferenceType(main);

        var pass = new PrimFuncPass { Name = "FoldLet" };
        pass.Add<FoldLet>();
        pass.Add<FoldConstCall>();
        pass.Add<FlattenSequential>();
        var new_func = await pass.RunAsync(main, new());
        Assert.True(new_func.Body[0] is TIR.For for1
                    && for1.Body[0] is Call);
    }

    [Fact]
    public async Task TestFoldLet2()
    {
        var main = T.PrimFunc("main", Callable.StackVMModuleKind).Body(// (*i8) -> ()
            T.Let(out var tcu_h_chunk, IR.F.Math.Min(10 + 9, 32) - 10).Body(
              T.Let(out var n_active_tcu, IR.F.Tensors.Cast(IR.F.Math.Ceil(48.0f / IR.F.Tensors.Cast(tcu_h_chunk, DataTypes.Float32)), DataTypes.Int32)).Body(
                T.If(IR.F.Math.Equal(n_active_tcu, 1)).Then(
                  new Call(new ExtraW(), 123))
                .Else(
                  new Call(new ExtraW(), 456)))))
        .Build();

        CompilerServices.InferenceType(main);

        var pass = new PrimFuncPass { Name = "FoldLet" };
        pass.Add<FoldLet>();
        pass.Add<FoldConstCall>();
        pass.Add<FlattenSequential>();
        pass.Add<FoldIfThen>();
        var new_func = await pass.RunAsync(main, new());
        Assert.True(new_func.Body[0] is Call { Target: ExtraW } call && call.Arguments[0].Evaluate().AsTensor().ToScalar<int>() == 456);
    }

    [Fact]
    public async Task TestFoldBufferIndex()
    {
        T.CreateBuffer(new(DataTypes.BFloat16, new[] { 3, 16, 24, 24 }), MemoryLocation.Input, out var ddr_if);
        T.CreateBuffer(new(DataTypes.BFloat16, new[] { 3, 16, 24, 24 }), MemoryLocation.Output, out var ddr_of);
        T.CreateBuffer(new(DataTypes.BFloat16, new[] { 3, 10, 5, 9 }), MemoryLocation.Data, out var glb_if);
        var bufferIndexMap = new Dictionary<TIR.Buffer, int>() {
          { ddr_if, 2 },
          { ddr_of, 4 },
        };

        PrimFunction main;
        {
            main = T.PrimFunc("main", Callable.StackVMModuleKind, ddr_if, ddr_of).Body(
             T.Unrolled(out var n, (0, 3, 3)).Body(
               T.Unrolled(out var c, (0, 16, 10)).Body(
                 T.Unrolled(out var h, (0, 24, 5)).Body(
                   T.Unrolled(out var w, (0, 24, 9)).Body(
                     new Call(new ExtraW(), IR.F.Buffer.BufferIndexOf(ddr_if)),
                     new Call(new ExtraW(), IR.F.Buffer.BufferIndexOf(ddr_of)))))))
           .Build();
        }

        var pass = new PrimFuncPass { Name = "AssginBuffer" };
        pass.Add<UnRollLoopSequential>();
        pass.Add<Substitutor>(Expr? (Expr e) =>
        {
            if (e is Call { } call && call.Arguments[0] is Buffer physicalBuffer && bufferIndexMap.TryGetValue(physicalBuffer, out var index))
            {
                return index;
            }

            return null;
        });
        pass.Add<FlattenSequential>();
        var post = await pass.RunAsync(main, new());
        {
            var getIndex = (int i, ParameterInfo info) =>
            {
                var index = (TensorConst)((Call)post.Body.Fields[i])[info];
                return index.Value.ToScalar<int>();
            };
            int count = 0;
            for (int n = 0; n < 3; n += 3)
            {
                for (int c = 0; c < 16; c += 10)
                {
                    for (int h = 0; h < 24; h += 5)
                    {
                        for (int w = 0; w < 24; w += 9)
                        {
                            Assert.Equal(2, getIndex(count++, ExtraW.Input));
                            Assert.Equal(4, getIndex(count++, ExtraW.Input));
                        }
                    }
                }
            }
        }
    }
}
