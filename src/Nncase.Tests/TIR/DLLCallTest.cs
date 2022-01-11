using Xunit;
using System;
using System.IO;
using Nncase.IR;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Linq;
using System.Linq.Expressions;
using System.Reflection;
using System.Reflection.Emit;

namespace Nncase.Tests.TIRTest
{
    public class DLLCallTest
    {

        public class CustomType
        {
            public delegate float declf(float x, float y);
        }

        [Fact]
        public void TestGetDelegate()
        {
            var cls_type = typeof(CustomType);
            Assert.Null(cls_type.GetField("declf"));
            Assert.NotNull(cls_type.GetMember("declf"));
            Assert.NotNull(cls_type.GetNestedType("declf"));
            var t = cls_type.GetNestedType("declf");
            Assert.Equal(typeof(MulticastDelegate), t.BaseType);
            ConstructorInfo ctor = t.GetConstructors()[0];
            Console.Write(ctor.GetMethodImplementationFlags());
            Console.Write(t.BaseType);
        }

        [Fact]
        public void TestDynamicBuildType()
        {
            AssemblyName aName = new AssemblyName("DynamicAssemblyExample");
            AssemblyBuilder ab = AssemblyBuilder.DefineDynamicAssembly(aName, AssemblyBuilderAccess.RunAndCollect);
            // For a single-module assembly, the module name is usually
            // the assembly name plus an extension.
            ModuleBuilder mb = ab.DefineDynamicModule(aName.Name);
            TypeBuilder tb = mb.DefineType("MyDynamicType", TypeAttributes.Public);
            // Add a private field of delegate

            var created_class = tb.CreateType();

            Console.WriteLine(created_class.GetMember("delfunc"));
        }

        public Type GetDynamicDeleType()
        {
            AssemblyName aName = new AssemblyName("DynamicAssemblyExample");
            AssemblyBuilder ab = AssemblyBuilder.DefineDynamicAssembly(aName, AssemblyBuilderAccess.RunAndCollect);
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
            return tb.CreateType();
        }

        [Fact]
        public void TestSimpleAdd()
        {

            var arch = RuntimeInformation.OSArchitecture == Architecture.Arm64 ? "arm64" : "x86_64";
            if (!RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
            {
                return;
            }
            var path = Testing.GetDumpDirPath("DLLCallTest/TestSimpleAdd");
            var src_path = Path.Combine(path, "main.c");
            var lib_path = Path.Combine(path, "main.dylib");
            using (var file = File.Open(src_path, FileMode.OpenOrCreate, FileAccess.Write))
            {
                using (var writer = new StreamWriter(file))
                {
                    writer.Write(@"
float fadd(float a, float b)
{
  return a + b;
}");
                }
            }
            var p = Process.Start("gcc", $"{src_path} -fPIC -shared -arch {arch} -o {lib_path}");

            var lib_ptr = NativeLibrary.Load(lib_path);
            var func_ptr = NativeLibrary.GetExport(lib_ptr, "fadd");

            var ctype = typeof(CustomType);
            var mtype = ctype.GetNestedType("declf");
            var func = Marshal.GetDelegateForFunctionPointer(func_ptr, mtype);
            var r = func.DynamicInvoke(1, 2);
            Assert.Equal(3.0f, r);

            var dy_dele = GetDynamicDeleType();
            var func2 = Marshal.GetDelegateForFunctionPointer(func_ptr, dy_dele);
            var r2 = func2.DynamicInvoke(2, 3);
            Assert.Equal(5.0f, r2);
        }
    }

}