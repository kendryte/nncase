using System;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using Nncase.Runtime;
using Nncase.Schedule;
using Xunit;


namespace Nncase.Tests.RuntimeTest
{


    // public class Interpreter
    // {
    //     public struct cRuntimeTensor
    //     {
    //         public IntPtr impl;
    //     }


    //     public struct cMemoryRange
    //     {
    //         MemoryLocation memory_location;
    //         ElemType datatype;
    //         UInt16 shared_module;
    //         uint start;
    //         uint size;
    //     }

    //     public delegate void DelInit();
    //     public delegate void DelLoadModel(byte[] buffer_ptr, int size);
    //     public delegate ulong DelGetSize();
    //     public delegate cMemoryRange DelGetDesc();
    //     public delegate cRuntimeTensor DelGetTensor();
    //     public delegate void DelSetTensor(ulong index, cRuntimeTensor rt);
    //     public delegate void DelRun();

    //     DelInit _init;
    //     DelLoadModel _load_model;
    //     DelGetSize _get_input_size, _get_output_size;
    //     DelGetDesc _get_input_desc, _get_output_desc;
    //     DelGetTensor _get_input_tensor, _get_output_tensor;
    //     DelSetTensor _set_input_tensor, _set_output_tensor;
    //     DelRun _run;

    //     IntPtr dllPtr;

    //     /// <summary>
    //     /// binding the exported function to the delegate
    //     /// </summary>
    //     /// <typeparam name="T"></typeparam>
    //     /// <param name="function_name"></param>
    //     /// <param name="dele"></param>
    //     void Binding<T>(string function_name, ref T dele)
    //       where T : Delegate
    //     {
    //         var funcPtr = NativeLibrary.GetExport(dllPtr, function_name);
    //         dele = Marshal.GetDelegateForFunctionPointer<T>(funcPtr);
    //     }

    //     public Interpreter(string dllPath)
    //     {
    //         dllPtr = NativeLibrary.Load(dllPath);

    //         Binding<DelInit>("interpreter_init", ref _init);
    //         Binding<DelLoadModel>("interpreter_load_model", ref _load_model);
    //         Binding<DelGetSize>("interpreter_inputs_size", ref _get_input_size);
    //         Binding<DelGetSize>("interpreter_outputs_size", ref _get_output_size);
    //         Binding<DelGetDesc>("interpreter_get_input_desc", ref _get_input_desc);
    //         Binding<DelGetDesc>("interpreter_get_output_desc", ref _get_output_desc);
    //         Binding<DelGetTensor>("interpreter_get_input_tensor", ref _get_input_tensor);
    //         Binding<DelSetTensor>("interpreter_set_input_tensor", ref _set_input_tensor);
    //         Binding<DelGetTensor>("interpreter_get_output_tensor", ref _get_output_tensor);
    //         Binding<DelSetTensor>("interpreter_set_output_tensor", ref _set_output_tensor);
    //         Binding<DelRun>("interpreter_run", ref _run);
    //         _init();
    //     }
    // }


    public class TestNncaseDll
    {

        [Fact]
        public void TestRuntimeTensorCreate()
        {
            var t = new DenseTensor<float>(new[] { 1, 2, 3, 4 });
            var rt = RuntimeTensor.FromDense(t);
        }


        [Fact]
        public void TestInterpreterInit()
        {
            // var interp = new Interpreter("/Users/lisa/Desktop/nncase/out/install/runtime-release/lib/libnncaseruntime_csharp.so");

        }

        [Fact]
        public void TestPinMame()
        {

        }

    }
}