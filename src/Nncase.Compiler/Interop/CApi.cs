// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Data;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.Quantization;
using Nncase.Runtime;
using Nncase.Runtime.Interop;
using static Nncase.Compiler.PythonHelper;

namespace Nncase.Compiler.Interop;

public enum ArrayElementKind
{
    /// <summary>
    /// <see cref="Runtime.Interop.RTValue"/>.
    /// </summary>
    RTValue = 0,
    Var = 1,
}

/// <summary>
/// Compiler C Api method table.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public unsafe struct CApiMT
{
    public delegate* unmanaged<ArrayElementKind, IntPtr*, nuint, IntPtr> ArrayCreatePtr;
    public delegate* unmanaged<IntPtr, nuint, IntPtr> ArrayGetItemPtr;
    public delegate* unmanaged<IntPtr, nuint> ArrayGetLengthPtr;
    public delegate* unmanaged<IntPtr, nuint, IntPtr, IntPtr> CalibrationDatasetProviderCreatePtr;
    public delegate* unmanaged<IntPtr, void> ClrHandleFreePtr;
    public delegate* unmanaged<IntPtr> CompileOptionsCreatePtr;
    public delegate* unmanaged<IntPtr, byte*, nuint, void> CompileOptionsSetInputFilePtr;
    public delegate* unmanaged<IntPtr, byte*, nuint, void> CompileOptionsSetInputFormatPtr;
    public delegate* unmanaged<IntPtr, byte*, nuint, void> CompileOptionsSetTargetPtr;
    public delegate* unmanaged<IntPtr, int, void> CompileOptionsSetDumpLevelPtr;
    public delegate* unmanaged<IntPtr, byte*, nuint, void> CompileOptionsSetDumpDirPtr;
    public delegate* unmanaged<IntPtr, IntPtr, void> CompileOptionsSetQuantizeOptionsPtr;
    public delegate* unmanaged<IntPtr, IntPtr, void> CompileOptionsSetQuantTypePtr;
    public delegate* unmanaged<IntPtr, ModelQuantMode, void> CompileOptionsSetModelQuantModePtr;
    public delegate* unmanaged<void> CompilerInitializePtr;
    public delegate* unmanaged<IntPtr, IntPtr> CompilerCreatePtr;
    public delegate* unmanaged<IntPtr, IntPtr, IntPtr> CompilerImportModulePtr;
    public delegate* unmanaged<IntPtr, void> CompilerCompilePtr;
    public delegate* unmanaged<IntPtr, IntPtr, void> CompilerGencodePtr;
    public delegate* unmanaged<Runtime.TypeCode, IntPtr> DataTypeFromTypeCodePtr;
    public delegate* unmanaged<IntPtr, IntPtr, IntPtr, IntPtr> ExprEvaluatePtr;
    public delegate* unmanaged<IntPtr, IntPtr> FunctionGetBodyPtr;
    public delegate* unmanaged<IntPtr, IntPtr> FunctionGetParametersPtr;
    public delegate* unmanaged<IntPtr, IntPtr> IRModuleGetEntryPtr;
    public delegate* unmanaged<void> LaunchDebuggerPtr;
    public delegate* unmanaged<IntPtr> QuantizeOptionsCreatePtr;
    public delegate* unmanaged<IntPtr, IntPtr, void> QuantizeOptionsSetCalibrationDatasetPtr;
    public delegate* unmanaged<IntPtr, CalibMethod, void> QuantizeOptionsSetCalibrationMethodPtr;
    public delegate* unmanaged<IntPtr, IntPtr> RTValueFromHandlePtr;
    public delegate* unmanaged<IntPtr, IntPtr> RTValueGetHandlePtr;
    public delegate* unmanaged<CStreamMT*, IntPtr, IntPtr> StreamCreatePtr;
    public delegate* unmanaged<byte*, nuint, byte> TargetExistsPtr;
}

/// <summary>
/// Compiler C Api.
/// </summary>
public static unsafe class CApi
{
    [UnmanagedCallersOnly]
    public static void Initialize(CApiMT* mt)
    {
        mt->ArrayCreatePtr = &ArrayCreate;
        mt->ArrayGetItemPtr = &ArrayGetItem;
        mt->ArrayGetLengthPtr = &ArrayGetLength;
        mt->CalibrationDatasetProviderCreatePtr = &CalibrationDatasetProviderCreate;
        mt->ClrHandleFreePtr = &ClrHandleFree;
        mt->CompileOptionsCreatePtr = &CompileOptionsCreate;
        mt->CompileOptionsSetInputFilePtr = &CompileOptionsSetInputFile;
        mt->CompileOptionsSetInputFormatPtr = &CompileOptionsSetInputFormat;
        mt->CompileOptionsSetTargetPtr = &CompileOptionsSetTarget;
        mt->CompileOptionsSetDumpLevelPtr = &CompileOptionsSetDumpLevel;
        mt->CompileOptionsSetDumpDirPtr = &CompileOptionsSetDumpDir;
        mt->CompileOptionsSetQuantizeOptionsPtr = &CompileOptionsSetQuantizeOptions;
        mt->CompileOptionsSetQuantTypePtr = &CompileOptionsSetQuantType;
        mt->CompileOptionsSetModelQuantModePtr = &CompileOptionsSetModelQuantMode;
        mt->CompilerInitializePtr = &CompilerInitialize;
        mt->CompilerCreatePtr = &CompilerCreate;
        mt->CompilerImportModulePtr = &CompilerImportModule;
        mt->CompilerCompilePtr = &CompilerCompile;
        mt->CompilerGencodePtr = &CompilerGencode;
        mt->DataTypeFromTypeCodePtr = &DataTypeFromTypeCode;
        mt->ExprEvaluatePtr = &ExprEvaluate;
        mt->FunctionGetBodyPtr = &FunctionGetBody;
        mt->FunctionGetParametersPtr = &FunctionGetParameters;
        mt->IRModuleGetEntryPtr = &IRModuleGetEntry;
        mt->LaunchDebuggerPtr = &LaunchDebugger;
        mt->QuantizeOptionsCreatePtr = &QuantizeOptionsCreate;
        mt->QuantizeOptionsSetCalibrationDatasetPtr = &QuantizeOptionsSetCalibrationDataset;
        mt->QuantizeOptionsSetCalibrationMethodPtr = &QuantizeOptionsSetCalibrationMethod;
        mt->RTValueFromHandlePtr = &RTValueFromHandle;
        mt->RTValueGetHandlePtr = &RTValueGetHandle;
        mt->StreamCreatePtr = &StreamCreate;
        mt->TargetExistsPtr = &TargetExists;
    }

    [UnmanagedCallersOnly]
    private static IntPtr ArrayCreate(ArrayElementKind kind, IntPtr* elements, nuint length)
    {
        return kind switch
        {
            ArrayElementKind.RTValue => ArrayCreateImpl<RTValue>(elements, length),
            ArrayElementKind.Var => ArrayCreateImpl<Var>(elements, length),
            _ => IntPtr.Zero,
        };
    }

    private static IntPtr ArrayCreateImpl<T>(IntPtr* elements, nuint length)
    {
        var array = new T[length];
        for (nuint i = 0; i < length; i++)
        {
            array[i] = Get<T>(elements[i]);
        }

        return GCHandle.ToIntPtr(GCHandle.Alloc(array));
    }

    [UnmanagedCallersOnly]
    private static IntPtr ArrayGetItem(IntPtr arrayHandle, nuint index)
    {
        var array = Get<Array>(arrayHandle);
        return GCHandle.ToIntPtr(GCHandle.Alloc(array.GetValue((long)index)));
    }

    [UnmanagedCallersOnly]
    private static nuint ArrayGetLength(IntPtr arrayHandle)
    {
        var array = Get<Array>(arrayHandle);
        return (nuint)array.LongLength;
    }

    [UnmanagedCallersOnly]
    private static IntPtr CalibrationDatasetProviderCreate(IntPtr datasetHandle, nuint samplesCount, IntPtr fnParamsHandle)
    {
        var dataset = Get<RTValue[]>(datasetHandle);
        var fnParams = Get<Var[]>(fnParamsHandle);
        if (dataset.Length != fnParams.Length * (int)samplesCount)
        {
            throw new ArgumentException($"Dataset count {dataset.Length} not equals to params count {fnParams.Length} * samples count {samplesCount}");
        }

        var samples = (dataset.Length == 0 ?
            Array.Empty<Dictionary<Var, IValue>>() :
            dataset.Chunk(dataset.Length).Select(inputs => inputs.Zip(fnParams).ToDictionary(
            item => item.Item2,
            item => item.Item1.ToValue()))).ToAsyncEnumerable();
        return GCHandle.ToIntPtr(GCHandle.Alloc(new CCalibrationDatasetProvider(samples, (int)samplesCount)));
    }

    [UnmanagedCallersOnly]
    private static void ClrHandleFree(IntPtr handle)
    {
        if (handle != IntPtr.Zero)
        {
            GCHandle.FromIntPtr(handle).Free();
        }
    }

    [UnmanagedCallersOnly]
    private static IntPtr CompileOptionsCreate()
    {
        return GCHandle.ToIntPtr(GCHandle.Alloc(new CompileOptions()));
    }

    [UnmanagedCallersOnly]
    private static void CompileOptionsSetInputFile(IntPtr compileOptionsHandle, byte* inputFilePtr, nuint inputFileLength)
    {
        Get<CompileOptions>(compileOptionsHandle).InputFile = ToString(inputFilePtr, inputFileLength);
    }

    [UnmanagedCallersOnly]
    private static void CompileOptionsSetInputFormat(IntPtr compileOptionsHandle, byte* inputFormatPtr, nuint inputFormatLength)
    {
        Get<CompileOptions>(compileOptionsHandle).InputFormat = ToString(inputFormatPtr, inputFormatLength);
    }

    [UnmanagedCallersOnly]
    private static void CompileOptionsSetTarget(IntPtr compileOptionsHandle, byte* targetPtr, nuint targetLength)
    {
        Get<CompileOptions>(compileOptionsHandle).Target = ToString(targetPtr, targetLength);
    }

    [UnmanagedCallersOnly]
    private static void CompileOptionsSetDumpLevel(IntPtr compileOptionsHandle, int dumpLevel)
    {
        Get<CompileOptions>(compileOptionsHandle).DumpLevel = dumpLevel;
    }

    [UnmanagedCallersOnly]
    private static void CompileOptionsSetDumpDir(IntPtr compileOptionsHandle, byte* dumpDirPtr, nuint dumpDirLength)
    {
        Get<CompileOptions>(compileOptionsHandle).DumpDir = ToString(dumpDirPtr, dumpDirLength);
    }

    [UnmanagedCallersOnly]
    private static void CompileOptionsSetQuantizeOptions(IntPtr compileOptionsHandle, IntPtr quantizeOptionsHandle)
    {
        Get<CompileOptions>(compileOptionsHandle).QuantizeOptions = Get<QuantizeOptions>(quantizeOptionsHandle);
    }

    [UnmanagedCallersOnly]
    private static void CompileOptionsSetQuantType(IntPtr compileOptionsHandle, IntPtr quantTypeHandle)
    {
        Get<CompileOptions>(compileOptionsHandle).QuantType = Get<DataType>(quantTypeHandle);
    }

    [UnmanagedCallersOnly]
    private static void CompileOptionsSetModelQuantMode(IntPtr compileOptionsHandle, ModelQuantMode quantMode)
    {
        Get<CompileOptions>(compileOptionsHandle).ModelQuantMode = quantMode;
    }

    [UnmanagedCallersOnly]
    private static void CompilerInitialize()
    {
        Compiler.Initialize();
    }

    [UnmanagedCallersOnly]
    private static IntPtr CompilerCreate(IntPtr compileOptionsHandle)
    {
        return GCHandle.ToIntPtr(GCHandle.Alloc(new Compiler(Get<CompileOptions>(compileOptionsHandle))));
    }

    [UnmanagedCallersOnly]
    private static IntPtr CompilerImportModule(IntPtr compilerHandle, IntPtr streamHandle)
    {
        var compiler = Get<Compiler>(compilerHandle);
        var stream = Get<CStream>(streamHandle);
        var module = compiler.ImportModule(stream);
        return GCHandle.ToIntPtr(GCHandle.Alloc(module));
    }

    [UnmanagedCallersOnly]
    private static void CompilerCompile(IntPtr compilerHandle)
    {
        var compiler = Get<Compiler>(compilerHandle);
        compiler.Compile();
    }

    [UnmanagedCallersOnly]
    private static void CompilerGencode(IntPtr compilerHandle, IntPtr streamHandle)
    {
        var compiler = Get<Compiler>(compilerHandle);
        var stream = Get<CStream>(streamHandle);
        compiler.Gencode(stream);
    }

    [UnmanagedCallersOnly]
    private static IntPtr DataTypeFromTypeCode(Runtime.TypeCode typeCode)
    {
        return GCHandle.ToIntPtr(GCHandle.Alloc(DataType.FromTypeCode(typeCode)));
    }

    [UnmanagedCallersOnly]
    private static IntPtr ExprEvaluate(IntPtr exprHandle, IntPtr fnParamsHandle, IntPtr inputsHandle)
    {
        var expr = Get<Expr>(exprHandle);
        var fnParams = Get<Var[]>(fnParamsHandle);
        var inputs = Get<RTValue[]>(inputsHandle);
        var result = CompilerServices.Evaluate(expr, fnParams.Zip(inputs).ToDictionary(
            x => x.Item1,
            x => x.Item2.ToValue()));
        var rtValue = RTValue.FromValue(result);
        return GCHandle.ToIntPtr(GCHandle.Alloc(rtValue));
    }

    [UnmanagedCallersOnly]
    private static IntPtr FunctionGetBody(IntPtr functionHandle)
    {
        var function = Get<Function>(functionHandle);
        return GCHandle.ToIntPtr(GCHandle.Alloc(function.Body));
    }

    [UnmanagedCallersOnly]
    private static IntPtr FunctionGetParameters(IntPtr functionHandle)
    {
        var function = Get<Function>(functionHandle);
        return GCHandle.ToIntPtr(GCHandle.Alloc(function.Parameters.ToArray()));
    }

    [UnmanagedCallersOnly]
    private static IntPtr IRModuleGetEntry(IntPtr moduleHandle)
    {
        var module = Get<IRModule>(moduleHandle);
        return GCHandle.ToIntPtr(GCHandle.Alloc(module.Entry));
    }

    [UnmanagedCallersOnly]
    private static void LaunchDebugger()
    {
        Debugger.Launch();
        while (!Debugger.IsAttached)
        {
            Thread.Yield();
        }
    }

    [UnmanagedCallersOnly]
    private static IntPtr QuantizeOptionsCreate()
    {
        return GCHandle.ToIntPtr(GCHandle.Alloc(new QuantizeOptions()));
    }

    [UnmanagedCallersOnly]
    private static void QuantizeOptionsSetCalibrationDataset(IntPtr quantizeOptionsHandle, IntPtr calibrationDatasetHandle)
    {
        Get<QuantizeOptions>(quantizeOptionsHandle).CalibrationDataset = Get<ICalibrationDatasetProvider>(calibrationDatasetHandle);
    }

    [UnmanagedCallersOnly]
    private static void QuantizeOptionsSetCalibrationMethod(IntPtr quantizeOptionsHandle, CalibMethod calibMethod)
    {
        Get<QuantizeOptions>(quantizeOptionsHandle).CalibrationMethod = calibMethod;
    }

    [UnmanagedCallersOnly]
    private static IntPtr RTValueFromHandle(IntPtr handle)
    {
        var rtValue = RTValue.FromHandle(handle);
        return GCHandle.ToIntPtr(GCHandle.Alloc(rtValue));
    }

    [UnmanagedCallersOnly]
    private static IntPtr RTValueGetHandle(IntPtr handle)
    {
        var rtValue = Get<RTValue>(handle);
        return rtValue.DangerousGetHandle();
    }

    [UnmanagedCallersOnly]
    private static IntPtr StreamCreate(CStreamMT* mt, IntPtr handle)
    {
        return GCHandle.ToIntPtr(GCHandle.Alloc(new CStream(mt, handle)));
    }

    [UnmanagedCallersOnly]
    private static byte TargetExists(byte* targetNamePtr, nuint targetNameLength)
    {
        var targetName = ToString(targetNamePtr, targetNameLength);
        try
        {
            CompilerServices.GetTarget(targetName);
            return 1;
        }
        catch
        {
            return 0;
        }
    }

    private static T Get<T>(IntPtr handle)
    {
        return (T)(GCHandle.FromIntPtr(handle).Target ?? throw new ArgumentNullException(nameof(handle)));
    }

    private static string ToString(byte* bytes, nuint length) =>
        Encoding.UTF8.GetString(bytes, (int)length);

    private class CCalibrationDatasetProvider : ICalibrationDatasetProvider
    {
        public CCalibrationDatasetProvider(IAsyncEnumerable<IReadOnlyDictionary<Var, IValue>> samples, int samplesCount)
        {
            Samples = samples;
            Count = samplesCount;
        }

        public int? Count { get; }

        public IAsyncEnumerable<IReadOnlyDictionary<Var, IValue>> Samples { get; }
    }
}
