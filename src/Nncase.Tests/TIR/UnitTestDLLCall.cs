// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Linq.Expressions;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.InteropServices;
using Nncase.IR;
using Nncase.TIR;
using Xunit;

namespace Nncase.Tests.TIRTest;

public class UnitTestDLLCall
{
    [Fact]
    public void TestGetDelegate()
    {
        var cls_type = typeof(CustomType);
        Assert.Null(cls_type.GetField("Declf"));
        Assert.NotNull(cls_type.GetMember("Declf"));
        Assert.NotNull(cls_type.GetNestedType("Declf"));
        var t = cls_type.GetNestedType("Declf");
        Assert.NotNull(t);
        Assert.Equal(typeof(MulticastDelegate), t.BaseType);
        ConstructorInfo ctor = t.GetConstructors()[0];
        Console.Write(ctor.GetMethodImplementationFlags());
        Console.Write(t.BaseType);
    }

    [Fact]
    public void TestDynamicBuildType()
    {
        var aName = new AssemblyName("DynamicAssemblyExample");
        var ab = AssemblyBuilder.DefineDynamicAssembly(aName, AssemblyBuilderAccess.RunAndCollect);

        // For a single-module assembly, the module name is usually
        // the assembly name plus an extension.
        Assert.NotNull(aName.Name);
        ModuleBuilder mb = ab.DefineDynamicModule(aName.Name);
        TypeBuilder tb = mb.DefineType("MyDynamicType", TypeAttributes.Public);

        // Add a private field of delegate
        var created_class = tb.CreateType();
        Assert.NotNull(created_class);

        Console.WriteLine(created_class.GetMember("delfunc"));
    }

    public System.Type GetDynamicDeleType()
    {
        var aName = new AssemblyName("DynamicAssemblyExample");
        var ab = AssemblyBuilder.DefineDynamicAssembly(aName, AssemblyBuilderAccess.RunAndCollect);
        Assert.NotNull(aName.Name);
        ModuleBuilder mb = ab.DefineDynamicModule(aName.Name);
        TypeBuilder tb = mb.DefineType("MyDynamicType", TypeAttributes.Public | TypeAttributes.Sealed, typeof(MulticastDelegate));
        ConstructorBuilder ctor = tb.DefineConstructor(MethodAttributes.Public | MethodAttributes.HideBySig | MethodAttributes.SpecialName | MethodAttributes.RTSpecialName, CallingConventions.Standard | CallingConventions.HasThis, new[] { typeof(object), typeof(IntPtr) });
        ctor.SetImplementationFlags(MethodImplAttributes.Runtime | MethodImplAttributes.Managed);

        var invoke = tb.DefineMethod("Invoke", MethodAttributes.Public | MethodAttributes.Virtual | MethodAttributes.HideBySig | MethodAttributes.NewSlot, CallingConventions.Standard | CallingConventions.HasThis, typeof(float), new[] { typeof(float), typeof(float) });
        invoke.SetImplementationFlags(MethodImplAttributes.Runtime | MethodImplAttributes.Managed);

        var beginInvoke = tb.DefineMethod("BeginInvoke", MethodAttributes.Public | MethodAttributes.Virtual | MethodAttributes.HideBySig | MethodAttributes.NewSlot, CallingConventions.Standard | CallingConventions.HasThis, typeof(IAsyncResult), new[] { typeof(float), typeof(float), typeof(IAsyncResult), typeof(object) });
        beginInvoke.SetImplementationFlags(MethodImplAttributes.Runtime | MethodImplAttributes.Managed);

        var endInvoke = tb.DefineMethod("EndInvoke", MethodAttributes.Public | MethodAttributes.Virtual | MethodAttributes.HideBySig | MethodAttributes.NewSlot, CallingConventions.Standard | CallingConventions.HasThis, typeof(float), new[] { typeof(IAsyncResult) });
        endInvoke.SetImplementationFlags(MethodImplAttributes.Runtime | MethodImplAttributes.Managed);
        var type = tb.CreateType();
        Assert.NotNull(type);
        return type;
    }

    public class CustomType
    {
        public delegate float Declf(float x, float y);
    }

    // [Fact]
    //     public void TestSimpleAdd()
    //     {

    // var arch = RuntimeInformation.OSArchitecture == Architecture.Arm64 ? "arm64" : "x86_64";
    //         if (!RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
    //         {
    //             return;
    //         }
    //         var path = Testing.GetDumpDirPath("DLLCallTest/TestSimpleAdd");
    //         var src_path = Path.Combine(path, "main.c");
    //         var lib_path = Path.Combine(path, "main.dylib");
    //         using (var file = File.Open(src_path, FileMode.OpenOrCreate, FileAccess.Write))
    //         {
    //             using (var writer = new StreamWriter(file))
    //             {
    //                 writer.Write(@"
    // float fadd(float a, float b)
    // {
    //   return a + b;
    // }");
    //             }
    //         }
    //         var p = Process.Start("clang", $"{src_path} -fPIC -shared -arch {arch} -o {lib_path}");

    // var lib_ptr = NativeLibrary.Load(lib_path);
    //         var func_ptr = NativeLibrary.GetExport(lib_ptr, "fadd");

    // var ctype = typeof(CustomType);
    //         var mtype = ctype.GetNestedType("declf");
    //         var func = Marshal.GetDelegateForFunctionPointer(func_ptr, mtype);
    //         var r = func.DynamicInvoke(1, 2);
    //         Assert.Equal(3.0f, r);

    // var dy_dele = GetDynamicDeleType();
    //         var func2 = Marshal.GetDelegateForFunctionPointer(func_ptr, dy_dele);
    //         var r2 = func2.DynamicInvoke(2, 3);
    //         Assert.Equal(5.0f, r2);
    //     }
}
