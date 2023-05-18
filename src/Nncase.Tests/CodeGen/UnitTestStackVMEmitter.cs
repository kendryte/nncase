// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Microsoft.VisualBasic;
using Nncase;
using Nncase.CodeGen;
using Nncase.CodeGen.StackVM;
using Nncase.IR;
using Nncase.Utilities;
using Xunit;

namespace Nncase.Tests.CoreTest;

public class UnitTestStackVMEmitter
{
    [Fact]
    public void TestStackVMEmitterGBr()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.Br(1);
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 90, 1, 0, 0, 0 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGAdd()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.Add();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 56 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGAnd()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.And();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 63 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGBreak()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.Break();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 99 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGCall()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.Call(1, 1);
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 94, 1, 0, 1, 0, 0, 0 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGCeq()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.Ceq();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 74 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGCge()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.Cge();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 75 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGCgt()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.Cgt();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 77 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGCle()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.Cle();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 72 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGCne()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.Cne();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 79 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGDiv()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.Div();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 59 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGDup()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.Dup();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 46 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGLdarg()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.Ldarg(1);
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 39, 1, 0 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGLdarg0()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.Ldarg0();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 40 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGLdarg1()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.Ldarg1();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 41 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGLdarg2()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.Ldarg2();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 42 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGLdarg3()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.Ldarg3();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 43 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGLdarg4()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.Ldarg4();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 44 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGLdarg5()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.Ldarg5();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 45 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGMul()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.Mul();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 58 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGNeg()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.Neg();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 55 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGConvI()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.ConvI();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 83 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGBrTrue()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.BrTrue(0);
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 91, 0, 0, 0, 0 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGCgeU()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.CgeU();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 76 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGCgtU()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.CgtU();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 78 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGCleU()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.CleU();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 73 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGClt()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.Clt();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 70 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGCltU()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.CltU();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 71 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGConvBR2()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.ConvBR2();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 88 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGConvI1()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.ConvI1();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 80 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGConvI2()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.ConvI2();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 81 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGConvR4()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.ConvR4();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 89 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGConvU()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.ConvU();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 87 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGConvU1()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.ConvU1();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 84 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGConvU2()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.ConvU2();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 85 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGConvU4()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.ConvU4();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 86 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGConvI4()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.ConvI4();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 82 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGCusCall()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.CusCall(string.Empty, Array.Empty<byte>(), 1);
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 97, 0, 0, 0, 0, 0, 1, 0 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGDivU()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.DivU();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 60 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGECall()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.ECall(0);
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 95, 0, 0 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGLdcI4_0()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.LdcI4_0();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 3 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGLdcI4_1()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.LdcI4_1();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 4 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGLdcR4()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.LdcR4(0f);
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 5, 0, 0, 0, 0 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGLdelemBR2()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.LdelemBR2();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 31 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGLdelemI()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.LdelemI();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 26 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGLdelemI1()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.LdelemI1();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 23 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGLdelemI2()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.LdelemI2();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 24 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGLdelemI4()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.LdelemI4();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 25 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGLdelemR4()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.LdelemR4();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 32 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGLdelemU()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.LdelemU();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 30 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGLdelemU1()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.LdelemU1();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 27 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGLdelemU2()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.LdelemU2();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 28 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGLdelemU4()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.LdelemU4();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 29 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGLdindBR2()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.LdindBR2();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 14 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGLdindI()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.LdindI();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 9 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGLdindI1()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.LdindI1();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 6 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGLdindI2()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.LdindI2();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 7 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGLdindI4()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.LdindI4();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 8 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGLdindR4()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.LdindR4();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 15 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGLdindU()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.LdindU();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 13 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGLdindU1()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.LdindU1();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 10 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGLdindU2()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.LdindU2();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 11 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGLdindU4()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.LdindU4();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 12 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGLdTupleElem()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.LdTupleElem();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 50 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGNop()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.Nop();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 0 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGNot()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.Not();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 66 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGOr()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.Or();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 64 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGPop()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.Pop();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 47 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGRem()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.Rem();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 61 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGRemU()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.RemU();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 62 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGRet()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.Ret();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 93 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGShl()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.Shl();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 67 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGShr()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.Shr();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 68 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGShrU()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.ShrU();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 69 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGStelemBR2()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.StelemBR2();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 37 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGStelemI()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.StelemI();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 36 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGStelemI1()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.StelemI1();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 33 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGStelemI2()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.StelemI2();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 34 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGStelemI4()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.StelemI4();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 35 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGStelemR4()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.StelemR4();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 38 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGStindBR2()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.StindBR2();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 20 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGStindI()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.StindI();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 19 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGStindI1()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.StindI1();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 16 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGStindI2()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.StindI2();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 17 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGStindI4()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.StindI4();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 18 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGStindR4()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.StindR4();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 21 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGSub()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.Sub();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 57 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGThrow()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.Throw();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 98 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGXor()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        stackVmEmitter.Xor();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 65 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGBinary()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        var tensorEmitter = new StackVMEmitter.TensorEmitter(stackVmEmitter);
        tensorEmitter.Binary(BinaryOp.Add);
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 100, actual[1], 0, 0 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGBatchToSpace()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        var tensorEmitter = new StackVMEmitter.TensorEmitter(stackVmEmitter);
        tensorEmitter.BatchToSpace();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 100, actual[1], 0 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGBitcast()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        var tensorEmitter = new StackVMEmitter.TensorEmitter(stackVmEmitter);
        tensorEmitter.Bitcast(DataTypes.Boolean, DataTypes.Boolean);
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 100, actual[1], 0, 0, 0 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGBroadcast()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        var tensorEmitter = new StackVMEmitter.TensorEmitter(stackVmEmitter);
        tensorEmitter.Broadcast();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 100, actual[1], 0 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGCast()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        var tensorEmitter = new StackVMEmitter.TensorEmitter(stackVmEmitter);
        tensorEmitter.Cast(DataTypes.Float32, CastMode.Exact);
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 100, actual[1], 0, 11, 1, 0, 0, 0 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGCelu()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        var tensorEmitter = new StackVMEmitter.TensorEmitter(stackVmEmitter);
        tensorEmitter.Celu();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 100, actual[1], 0 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGClamp()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        var tensorEmitter = new StackVMEmitter.TensorEmitter(stackVmEmitter);
        tensorEmitter.Clamp();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 100, actual[1], 0 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGCompare()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        var tensorEmitter = new StackVMEmitter.TensorEmitter(stackVmEmitter);
        tensorEmitter.Compare(CompareOp.Equal);
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 100, actual[1], 0, 0 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGConcat()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        var tensorEmitter = new StackVMEmitter.TensorEmitter(stackVmEmitter);
        tensorEmitter.Concat();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 100, actual[1], 0 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGCondition()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        var tensorEmitter = new StackVMEmitter.TensorEmitter(stackVmEmitter);
        tensorEmitter.Condition(true);
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 100, actual[1], 0, 1 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGConstantOfShape()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        var tensorEmitter = new StackVMEmitter.TensorEmitter(stackVmEmitter);
        tensorEmitter.ConstantOfShape();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 100, actual[1], 0 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGConv2D()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        var tensorEmitter = new StackVMEmitter.TensorEmitter(stackVmEmitter);
        tensorEmitter.Conv2D(PadMode.Constant);
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 100, actual[1], 0, 0 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGConv2DTranspose()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        var tensorEmitter = new StackVMEmitter.TensorEmitter(stackVmEmitter);
        tensorEmitter.Conv2DTranspose(PadMode.Constant);
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 100, actual[1], 0, 0 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGCumSum()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        var tensorEmitter = new StackVMEmitter.TensorEmitter(stackVmEmitter);
        tensorEmitter.CumSum();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 100, actual[1], 0 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGDequantize()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        var tensorEmitter = new StackVMEmitter.TensorEmitter(stackVmEmitter);
        tensorEmitter.Dequantize(DataTypes.Boolean);
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 100, actual[1], 0, 0 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGBatchNormalization()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        var tensorEmitter = new StackVMEmitter.TensorEmitter(stackVmEmitter);
        tensorEmitter.BatchNormalization();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 100, actual[1], 0 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGElu()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        var tensorEmitter = new StackVMEmitter.TensorEmitter(stackVmEmitter);
        tensorEmitter.Elu();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 100, actual[1], 0 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGErf()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        var tensorEmitter = new StackVMEmitter.TensorEmitter(stackVmEmitter);
        tensorEmitter.Erf();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 100, actual[1], 0 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGExpand()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        var tensorEmitter = new StackVMEmitter.TensorEmitter(stackVmEmitter);
        tensorEmitter.Expand();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 100, actual[1], 0 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGFakeDequantize()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        var tensorEmitter = new StackVMEmitter.TensorEmitter(stackVmEmitter);
        tensorEmitter.FakeDequantize(DataTypes.Boolean);
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 100, actual[1], 0, 0 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGFakeQuantize()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        var tensorEmitter = new StackVMEmitter.TensorEmitter(stackVmEmitter);
        tensorEmitter.FakeQuantize(DataTypes.Boolean);
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 100, actual[1], 0, 0 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGFlatten()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        var tensorEmitter = new StackVMEmitter.TensorEmitter(stackVmEmitter);
        tensorEmitter.Flatten();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 100, actual[1], 0 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGGather()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        var tensorEmitter = new StackVMEmitter.TensorEmitter(stackVmEmitter);
        tensorEmitter.Gather();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 100, actual[1], 0 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGGatherND()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        var tensorEmitter = new StackVMEmitter.TensorEmitter(stackVmEmitter);
        tensorEmitter.GatherND();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 100, actual[1], 0 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGGelu()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        var tensorEmitter = new StackVMEmitter.TensorEmitter(stackVmEmitter);
        tensorEmitter.Gelu();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 100, actual[1], 0 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGHardmax()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        var tensorEmitter = new StackVMEmitter.TensorEmitter(stackVmEmitter);
        tensorEmitter.Hardmax();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 100, actual[1], 0 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGHardSigmoid()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        var tensorEmitter = new StackVMEmitter.TensorEmitter(stackVmEmitter);
        tensorEmitter.HardSigmoid();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 100, actual[1], 0 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGHardSwish()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        var tensorEmitter = new StackVMEmitter.TensorEmitter(stackVmEmitter);
        tensorEmitter.HardSwish();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 100, actual[1], 0 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGInstanceNormalization()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        var tensorEmitter = new StackVMEmitter.TensorEmitter(stackVmEmitter);
        tensorEmitter.InstanceNormalization();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 100, actual[1], 0 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGL2Normalization()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        var tensorEmitter = new StackVMEmitter.TensorEmitter(stackVmEmitter);
        tensorEmitter.L2Normalization();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 100, actual[1], 0 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGLayerNorm()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        var tensorEmitter = new StackVMEmitter.TensorEmitter(stackVmEmitter);
        tensorEmitter.LayerNorm(-1, 0f);
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 100, actual[1], 0, 255, 255, 255, 255, 0, 0, 0, 0 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGLeakyRelu()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        var tensorEmitter = new StackVMEmitter.TensorEmitter(stackVmEmitter);
        tensorEmitter.LeakyRelu();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 100, actual[1], 0 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGLogSoftmax()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        var tensorEmitter = new StackVMEmitter.TensorEmitter(stackVmEmitter);
        tensorEmitter.LogSoftmax();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 100, actual[1], 0 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGLpNormalization()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        var tensorEmitter = new StackVMEmitter.TensorEmitter(stackVmEmitter);
        tensorEmitter.LpNormalization();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 100, actual[1], 0 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGLRN()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        var tensorEmitter = new StackVMEmitter.TensorEmitter(stackVmEmitter);
        tensorEmitter.LRN();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 100, actual[1], 0 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGLSTM()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        var tensorEmitter = new StackVMEmitter.TensorEmitter(stackVmEmitter);
        tensorEmitter.LSTM(LSTMDirection.Bidirectional, LSTMLayout.One, new[] { "tanh" });
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 100, actual[1], 0, 2, 0, 0, 0, 1, 0, 0, 0, 116, 97, 110, 104, 0, 0 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGMatMul()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        var tensorEmitter = new StackVMEmitter.TensorEmitter(stackVmEmitter);
        tensorEmitter.MatMul();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 100, actual[1], 0 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGNormal()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        var tensorEmitter = new StackVMEmitter.TensorEmitter(stackVmEmitter);
        tensorEmitter.Normal(DataTypes.Boolean);
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 100, actual[1], 0, 0 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGNormalLike()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        var tensorEmitter = new StackVMEmitter.TensorEmitter(stackVmEmitter);
        tensorEmitter.NormalLike(DataTypes.Boolean);
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 100, actual[1], 0, 0 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGOneHot()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        var tensorEmitter = new StackVMEmitter.TensorEmitter(stackVmEmitter);
        tensorEmitter.OneHot(OneHotMode.Normal);
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 100, actual[1], 0, 0 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGPad()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        var tensorEmitter = new StackVMEmitter.TensorEmitter(stackVmEmitter);
        tensorEmitter.Pad(PadMode.Constant);
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 100, actual[1], 0, 0 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGPRelu()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        var tensorEmitter = new StackVMEmitter.TensorEmitter(stackVmEmitter);
        tensorEmitter.PRelu();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 100, actual[1], 0 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGProd()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        var tensorEmitter = new StackVMEmitter.TensorEmitter(stackVmEmitter);
        tensorEmitter.Prod();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 100, actual[1], 0 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGQuantize()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        var tensorEmitter = new StackVMEmitter.TensorEmitter(stackVmEmitter);
        tensorEmitter.Quantize(DataTypes.Boolean);
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 100, actual[1], 0, 0 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGQuantParamOf()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        var tensorEmitter = new StackVMEmitter.TensorEmitter(stackVmEmitter);
        tensorEmitter.QuantParamOf(QuantMode.UnsignedMode);
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 100, actual[1], 0, 0, 0, 0, 0 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGRange()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        var tensorEmitter = new StackVMEmitter.TensorEmitter(stackVmEmitter);
        tensorEmitter.Range();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 100, actual[1], 0 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGRangeOf()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        var tensorEmitter = new StackVMEmitter.TensorEmitter(stackVmEmitter);
        tensorEmitter.RangeOf(false);
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 100, actual[1], 0, 0 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGReduce()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        var tensorEmitter = new StackVMEmitter.TensorEmitter(stackVmEmitter);
        tensorEmitter.Reduce(ReduceOp.Prod);
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 100, actual[1], 0, 4 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGReduceArg()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        var tensorEmitter = new StackVMEmitter.TensorEmitter(stackVmEmitter);
        tensorEmitter.ReduceArg(ReduceArgOp.ArgMax, DataTypes.Float32);
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 100, actual[1], 0, 1, 11 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGReduceWindow2D()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        var tensorEmitter = new StackVMEmitter.TensorEmitter(stackVmEmitter);
        tensorEmitter.ReduceWindow2D(ReduceOp.Prod);
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 100, actual[1], 0, 4 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGRelu()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        var tensorEmitter = new StackVMEmitter.TensorEmitter(stackVmEmitter);
        tensorEmitter.Relu();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 100, actual[1], 0 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGRelu6()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        var tensorEmitter = new StackVMEmitter.TensorEmitter(stackVmEmitter);
        tensorEmitter.Relu6();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 100, actual[1], 0 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGRequire()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        var tensorEmitter = new StackVMEmitter.TensorEmitter(stackVmEmitter);
        tensorEmitter.Require(string.Empty);
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 100, actual[1], 0, 0 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGReshape()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        var tensorEmitter = new StackVMEmitter.TensorEmitter(stackVmEmitter);
        tensorEmitter.Reshape();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 100, actual[1], 0 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGReverseSequence()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        var tensorEmitter = new StackVMEmitter.TensorEmitter(stackVmEmitter);
        tensorEmitter.ReverseSequence();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 100, actual[1], 0 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGResizeImage()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        var tensorEmitter = new StackVMEmitter.TensorEmitter(stackVmEmitter);
        tensorEmitter.ResizeImage(ImageResizeMode.Bilinear, ImageResizeTransformationMode.Asymmetric, ImageResizeNearestMode.Ceil, false);
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 100, actual[1], 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 0 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGSelect()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        var tensorEmitter = new StackVMEmitter.TensorEmitter(stackVmEmitter);
        tensorEmitter.Select();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 100, actual[1], 0 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGSelu()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        var tensorEmitter = new StackVMEmitter.TensorEmitter(stackVmEmitter);
        tensorEmitter.Selu();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 100, actual[1], 0 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGShapeOf()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        var tensorEmitter = new StackVMEmitter.TensorEmitter(stackVmEmitter);
        tensorEmitter.ShapeOf();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 100, actual[1], 0 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGSigmoid()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        var tensorEmitter = new StackVMEmitter.TensorEmitter(stackVmEmitter);
        tensorEmitter.Sigmoid();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 100, actual[1], 0 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGSizeOf()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        var tensorEmitter = new StackVMEmitter.TensorEmitter(stackVmEmitter);
        tensorEmitter.SizeOf();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 100, actual[1], 0 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGSlice()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        var tensorEmitter = new StackVMEmitter.TensorEmitter(stackVmEmitter);
        tensorEmitter.Slice();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 100, actual[1], 0 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGSoftmax()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        var tensorEmitter = new StackVMEmitter.TensorEmitter(stackVmEmitter);
        tensorEmitter.Softmax();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 100, actual[1], 0 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGSoftplus()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        var tensorEmitter = new StackVMEmitter.TensorEmitter(stackVmEmitter);
        tensorEmitter.Softplus();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 100, actual[1], 0 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGSpaceToBatch()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        var tensorEmitter = new StackVMEmitter.TensorEmitter(stackVmEmitter);
        tensorEmitter.SpaceToBatch();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 100, actual[1], 0 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGSplit()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        var tensorEmitter = new StackVMEmitter.TensorEmitter(stackVmEmitter);
        tensorEmitter.Split();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 100, actual[1], 0 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGSqueeze()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        var tensorEmitter = new StackVMEmitter.TensorEmitter(stackVmEmitter);
        tensorEmitter.Squeeze();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 100, actual[1], 0 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGStack()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        var tensorEmitter = new StackVMEmitter.TensorEmitter(stackVmEmitter);
        tensorEmitter.Stack();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 100, actual[1], 0 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGSwish()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        var tensorEmitter = new StackVMEmitter.TensorEmitter(stackVmEmitter);
        tensorEmitter.Swish();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 100, actual[1], 0 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGTile()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        var tensorEmitter = new StackVMEmitter.TensorEmitter(stackVmEmitter);
        tensorEmitter.Tile();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 100, actual[1], 0 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGTopK()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        var tensorEmitter = new StackVMEmitter.TensorEmitter(stackVmEmitter);
        tensorEmitter.TopK();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 100, actual[1], 0 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGTranspose()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        var tensorEmitter = new StackVMEmitter.TensorEmitter(stackVmEmitter);
        tensorEmitter.Transpose();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 100, actual[1], 0 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGUniform()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        var tensorEmitter = new StackVMEmitter.TensorEmitter(stackVmEmitter);
        tensorEmitter.Uniform(DataTypes.Float32);
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 100, actual[1], 0, 11 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGUniformLike()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        var tensorEmitter = new StackVMEmitter.TensorEmitter(stackVmEmitter);
        tensorEmitter.UniformLike(DataTypes.Float32);
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 100, actual[1], 0, 11 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGUnsqueeze()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        var tensorEmitter = new StackVMEmitter.TensorEmitter(stackVmEmitter);
        tensorEmitter.Unsqueeze();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 100, actual[1], 0 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGWhere()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        var tensorEmitter = new StackVMEmitter.TensorEmitter(stackVmEmitter);
        tensorEmitter.Where(false);
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 100, actual[1], 0, 0 }, actual);
    }

    [Fact]
    public void TestStackVMEmitterGSoftsign()
    {
        var memoryStream = new MemoryStream();
        var stackVmEmitter = new StackVMEmitter(new BinaryWriter(memoryStream, Encoding.UTF8, true));
        var tensorEmitter = new StackVMEmitter.TensorEmitter(stackVmEmitter);
        tensorEmitter.Softsign();
        var actual = memoryStream.ToArray();
        Assert.Equal(new byte[] { 100, actual[1], 0 }, actual);
    }
}
