// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.Compiler.Interop;

/// <summary>
/// Compiler C Api method table.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public unsafe struct CApiMT
{
    public delegate* unmanaged<IntPtr, void> ClrHandleFreePtr;
    public delegate* unmanaged<IntPtr> CompileOptionsCreatePtr;
    public delegate* unmanaged<IntPtr, byte*, nuint, void> CompileOptionsSetInputFilePtr;
    public delegate* unmanaged<IntPtr, byte*, nuint, void> CompileOptionsSetInputFormatPtr;
    public delegate* unmanaged<IntPtr, byte*, nuint, void> CompileOptionsSetTargetPtr;
    public delegate* unmanaged<IntPtr, int, void> CompileOptionsSetDumpLevelPtr;
    public delegate* unmanaged<IntPtr, byte*, nuint, void> CompileOptionsSetDumpDirPtr;
    public delegate* unmanaged<IntPtr, IntPtr, void> CompileOptionsSetQuantTypePtr;
    public delegate* unmanaged<IntPtr, QuantMode, void> CompileOptionsSetQuantModePtr;
    public delegate* unmanaged<void> CompilerInitializePtr;
    public delegate* unmanaged<IntPtr, IntPtr> CompilerCreatePtr;
    public delegate* unmanaged<IntPtr, IntPtr, IntPtr, IntPtr> CompilerImportModulePtr;
    public delegate* unmanaged<Runtime.TypeCode, IntPtr> DataTypeFromTypeCodePtr;
    public delegate* unmanaged<CStreamMT*, IntPtr, IntPtr> StreamCreatePtr;
}

/// <summary>
/// Compiler C Api.
/// </summary>
public static unsafe class CApi
{
    [UnmanagedCallersOnly]
    public static void Initialize(CApiMT* mt)
    {
        mt->ClrHandleFreePtr = &ClrHandleFree;
        mt->CompileOptionsCreatePtr = &CompileOptionsCreate;
        mt->CompileOptionsSetInputFilePtr = &CompileOptionsSetInputFile;
        mt->CompileOptionsSetInputFormatPtr = &CompileOptionsSetInputFormat;
        mt->CompileOptionsSetTargetPtr = &CompileOptionsSetTarget;
        mt->CompileOptionsSetDumpLevelPtr = &CompileOptionsSetDumpLevel;
        mt->CompileOptionsSetDumpDirPtr = &CompileOptionsSetDumpDir;
        mt->CompileOptionsSetQuantTypePtr = &CompileOptionsSetQuantType;
        mt->CompileOptionsSetQuantModePtr = &CompileOptionsSetQuantMode;
        mt->CompilerInitializePtr = &CompilerInitialize;
        mt->DataTypeFromTypeCodePtr = &DataTypeFromTypeCode;
        mt->StreamCreatePtr = &StreamCreate;
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
    private static void CompileOptionsSetQuantType(IntPtr compileOptionsHandle, IntPtr quantTypeHandle)
    {
        Get<CompileOptions>(compileOptionsHandle).QuantType = Get<DataType>(quantTypeHandle);
    }

    [UnmanagedCallersOnly]
    private static void CompileOptionsSetQuantMode(IntPtr compileOptionsHandle, QuantMode quantMode)
    {
        Get<CompileOptions>(compileOptionsHandle).QuantMode = quantMode;
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
    private static IntPtr CompilerImportModule(IntPtr compilerHandle, IntPtr streamHandle, IntPtr compileOptionsHandle)
    {
        var compiler = Get<Compiler>(compilerHandle);
        var stream = Get<CStream>(streamHandle);
        var compileOptions = Get<CompileOptions>(compileOptionsHandle);
        var module = compiler.ImportModule(stream, compileOptions);
        return GCHandle.ToIntPtr(GCHandle.Alloc(module));
    }

    [UnmanagedCallersOnly]
    private static IntPtr DataTypeFromTypeCode(Runtime.TypeCode typeCode)
    {
        return GCHandle.ToIntPtr(GCHandle.Alloc(DataType.FromTypeCode(typeCode)));
    }

    [UnmanagedCallersOnly]
    private static IntPtr StreamCreate(CStreamMT* mt, IntPtr handle)
    {
        return GCHandle.ToIntPtr(GCHandle.Alloc(new CStream(mt, handle)));
    }

    private static T Get<T>(IntPtr handle)
        where T : class
    {
        return (T)(GCHandle.FromIntPtr(handle).Target ?? throw new ArgumentNullException(nameof(handle)));
    }

    private static string ToString(byte* bytes, nuint length) =>
        Encoding.UTF8.GetString(bytes, (int)length);
}
