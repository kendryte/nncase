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
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Newtonsoft.Json;
using Nncase.Diagnostics;
using Nncase.Hosting;
using Nncase.IR;
using Nncase.Quantization;
using Nncase.Runtime;
using Nncase.Runtime.Interop;

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
    public delegate* unmanaged<IntPtr, void> ClrHandleDisposePtr;
    public delegate* unmanaged<IntPtr, void> ClrHandleFreePtr;
    public delegate* unmanaged<IntPtr> CompileOptionsCreatePtr;
    public delegate* unmanaged<IntPtr, byte*, nuint, void> CompileOptionsSetInputFilePtr;
    public delegate* unmanaged<IntPtr, byte*, nuint, void> CompileOptionsSetInputFormatPtr;
    public delegate* unmanaged<IntPtr, byte*, nuint, void> CompileOptionsSetDumpDirPtr;
    public delegate* unmanaged<IntPtr, DumpFlags> CompileOptionsGetDumpFlagsPtr;
    public delegate* unmanaged<IntPtr, DumpFlags, void> CompileOptionsSetDumpFlagsPtr;
    public delegate* unmanaged<IntPtr, IntPtr, void> CompileOptionsSetQuantizeOptionsPtr;
    public delegate* unmanaged<IntPtr, byte, void> CompileOptionsSetPreProcessPtr;
    public delegate* unmanaged<IntPtr, byte*, nuint, void> CompileOptionsSetInputLayoutPtr;
    public delegate* unmanaged<IntPtr, byte*, nuint, void> CompileOptionsSetOutputLayoutPtr;
    public delegate* unmanaged<IntPtr, byte, void> CompileOptionsSetInputTypePtr;
    public delegate* unmanaged<IntPtr, byte*, nuint, void> CompileOptionsSetInputShapePtr;
    public delegate* unmanaged<IntPtr, byte*, nuint, void> CompileOptionsSetInputRangePtr;
    public delegate* unmanaged<IntPtr, byte, void> CompileOptionsSetSwapRBPtr;
    public delegate* unmanaged<IntPtr, float, void> CompileOptionsSetLetterBoxValuePtr;
    public delegate* unmanaged<IntPtr, byte*, nuint, void> CompileOptionsSetMeanPtr;
    public delegate* unmanaged<IntPtr, byte*, nuint, void> CompileOptionsSetStdPtr;
    public delegate* unmanaged<IntPtr, IntPtr, void> CompileOptionsSetShapeBucketOptionsPtr;
    public delegate* unmanaged<IntPtr, IntPtr, void> CompileOptionsSetCpuTargetOptionsPtr;
    public delegate* unmanaged<IntPtr, IntPtr, IntPtr> CompileSessionCreatePtr;
    public delegate* unmanaged<IntPtr, IntPtr> CompileSessionGetCompilerPtr;
    public delegate* unmanaged<void> CompilerInitializePtr;
    public delegate* unmanaged<IntPtr, IntPtr, IntPtr> CompilerImportTFLiteModulePtr;
    public delegate* unmanaged<IntPtr, IntPtr, IntPtr> CompilerImportOnnxModulePtr;
    public delegate* unmanaged<IntPtr, IntPtr, IntPtr, IntPtr> CompilerImportNcnnModulePtr;
    public delegate* unmanaged<IntPtr, void> CompilerCompilePtr;
    public delegate* unmanaged<IntPtr, IntPtr, void> CompilerGencodePtr;
    public delegate* unmanaged<Runtime.TypeCode, IntPtr> DataTypeFromTypeCodePtr;
    public delegate* unmanaged<IntPtr, IntPtr, IntPtr, IntPtr> ExprEvaluatePtr;
    public delegate* unmanaged<IntPtr, IntPtr> FunctionGetBodyPtr;
    public delegate* unmanaged<IntPtr, IntPtr> FunctionGetParametersPtr;
    public delegate* unmanaged<IntPtr, IntPtr> IRModuleGetEntryPtr;
    public delegate* unmanaged<void> LaunchDebuggerPtr;
    public delegate* unmanaged<IntPtr> QuantizeOptionsCreatePtr;
    public delegate* unmanaged<IntPtr> ShapeBucketOptionsCreatePtr;
    public delegate* unmanaged<IntPtr> CpuTargetOptionsCreatePtr;
    public delegate* unmanaged<IntPtr, IntPtr, void> QuantizeOptionsSetCalibrationDatasetPtr;
    public delegate* unmanaged<IntPtr, CalibMethod, void> QuantizeOptionsSetCalibrationMethodPtr;
    public delegate* unmanaged<IntPtr, ModelQuantMode, void> QuantizeOptionsSetModelQuantModePtr;
    public delegate* unmanaged<IntPtr, QuantType, void> QuantizeOptionsSetQuantTypePtr;
    public delegate* unmanaged<IntPtr, QuantType, void> QuantizeOptionsSetWQuantTypePtr;
    public delegate* unmanaged<IntPtr, FineTuneWeightsMethod, void> QuantOptionsSetFineTuneWeightsMethodPtr;
    public delegate* unmanaged<IntPtr, byte, void> QuantOptionsSetUseMixQuantPtr;
    public delegate* unmanaged<IntPtr, byte*, nuint, void> QuantOptionsSetQuantSchemePtr;
    public delegate* unmanaged<IntPtr, byte, void> QuantOptionsSetQuantSchemeStrictModePtr;
    public delegate* unmanaged<IntPtr, byte, void> QuantOptionsSetExportQuantSchemePtr;
    public delegate* unmanaged<IntPtr, byte, void> QuantOptionsSetExportWeightRangeByChannelPtr;
    public delegate* unmanaged<IntPtr, byte, void> QuantOptionsSetDumpQuantErrorPtr;
    public delegate* unmanaged<IntPtr, byte, void> QuantOptionsSetDumpQuantErrorSymmetricForSignedPtr;
    public delegate* unmanaged<IntPtr, byte, void> ShapeBucketOptionsSetEnablePtr;
    public delegate* unmanaged<IntPtr, byte*, nuint, void> ShapeBucketOptionsSetRangeInfoPtr;
    public delegate* unmanaged<IntPtr, nuint, void> ShapeBucketOptionsSetSegmentsCountPtr;
    public delegate* unmanaged<IntPtr, byte*, nuint, void> ShapeBucketOptionsSetFixVarMapPtr;
    public delegate* unmanaged<IntPtr, byte, void> CpuTargetOptionsSetPackingPtr;
    public delegate* unmanaged<IntPtr, IntPtr> RTValueFromHandlePtr;
    public delegate* unmanaged<IntPtr, IntPtr> RTValueGetHandlePtr;
    public delegate* unmanaged<CStreamMT*, IntPtr, IntPtr> StreamCreatePtr;
    public delegate* unmanaged<byte*, nuint, IntPtr> TargetCreatePtr;
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
        mt->ClrHandleDisposePtr = &ClrHandleDispose;
        mt->ClrHandleFreePtr = &ClrHandleFree;
        mt->CompileOptionsCreatePtr = &CompileOptionsCreate;
        mt->CompileOptionsSetInputFilePtr = &CompileOptionsSetInputFile;
        mt->CompileOptionsSetInputFormatPtr = &CompileOptionsSetInputFormat;
        mt->CompileOptionsSetDumpDirPtr = &CompileOptionsSetDumpDir;
        mt->CompileOptionsSetDumpFlagsPtr = &CompileOptionsSetDumpFlags;
        mt->CompileOptionsGetDumpFlagsPtr = &CompileOptionsGetDumpFlags;
        mt->CompileOptionsSetQuantizeOptionsPtr = &CompileOptionsSetQuantizeOptions;
        mt->CompileOptionsSetPreProcessPtr = &CompileOptionsSetPreProcess;
        mt->CompileOptionsSetInputLayoutPtr = &CompileOptionsSetInputLayout;
        mt->CompileOptionsSetOutputLayoutPtr = &CompileOptionsSetOutputLayout;
        mt->CompileOptionsSetInputTypePtr = &CompileOptionsSetInputType;
        mt->CompileOptionsSetInputShapePtr = &CompileOptionsSetInputShape;
        mt->CompileOptionsSetInputRangePtr = &CompileOptionsSetInputRange;
        mt->CompileOptionsSetSwapRBPtr = &CompileOptionsSetSwapRB;
        mt->CompileOptionsSetLetterBoxValuePtr = &CompileOptionsSetLetterBoxValue;
        mt->CompileOptionsSetMeanPtr = &CompileOptionsSetMean;
        mt->CompileOptionsSetStdPtr = &CompileOptionsSetStd;
        mt->CompileOptionsSetShapeBucketOptionsPtr = &CompileOptionsSetShapeBucketOptions;
        mt->CompileOptionsSetCpuTargetOptionsPtr = &CompileOptionsSetCpuTargetOptions;
        mt->CompileSessionCreatePtr = &CompileSessionCreate;
        mt->CompileSessionGetCompilerPtr = &CompileSessionGetCompiler;
        mt->CompilerInitializePtr = &CompilerInitialize;
        mt->CompilerImportTFLiteModulePtr = &CompilerImportTFLiteModule;
        mt->CompilerImportOnnxModulePtr = &CompilerImportOnnxModule;
        mt->CompilerImportNcnnModulePtr = &CompilerImportNcnnModule;
        mt->CompilerCompilePtr = &CompilerCompile;
        mt->CompilerGencodePtr = &CompilerGencode;
        mt->DataTypeFromTypeCodePtr = &DataTypeFromTypeCode;
        mt->ExprEvaluatePtr = &ExprEvaluate;
        mt->FunctionGetBodyPtr = &FunctionGetBody;
        mt->FunctionGetParametersPtr = &FunctionGetParameters;
        mt->IRModuleGetEntryPtr = &IRModuleGetEntry;
        mt->LaunchDebuggerPtr = &LaunchDebugger;
        mt->QuantizeOptionsCreatePtr = &QuantizeOptionsCreate;
        mt->ShapeBucketOptionsCreatePtr = &ShapeBucketOptionsCreate;
        mt->CpuTargetOptionsCreatePtr = &CpuTargetOptionsCreate;
        mt->QuantizeOptionsSetCalibrationDatasetPtr = &QuantizeOptionsSetCalibrationDataset;
        mt->QuantizeOptionsSetCalibrationMethodPtr = &QuantizeOptionsSetCalibrationMethod;
        mt->QuantizeOptionsSetModelQuantModePtr = &QuantizeOptionsSetModelQuantMode;
        mt->QuantizeOptionsSetQuantTypePtr = &QuantizeOptionsSetQuantType;
        mt->QuantizeOptionsSetWQuantTypePtr = &QuantizeOptionsSetWQuantType;
        mt->QuantOptionsSetFineTuneWeightsMethodPtr = &QuantizeOptionsSetFineTuneWeightsMethod;
        mt->QuantOptionsSetUseMixQuantPtr = &QuantOptionsSetUseMixQuant;
        mt->QuantOptionsSetQuantSchemePtr = &QuantizeOptionsSetQuantScheme;
        mt->QuantOptionsSetQuantSchemeStrictModePtr = &QuantizeOptionsSetQuantSchemeStrictMode;
        mt->QuantOptionsSetExportQuantSchemePtr = &QuantizeOptionsSetExportQuantScheme;
        mt->QuantOptionsSetExportWeightRangeByChannelPtr = &QuantizeOptionsSetExportWeightRangeByChannel;
        mt->QuantOptionsSetDumpQuantErrorPtr = &QuantizeOptionsSetDumpQuantError;
        mt->QuantOptionsSetDumpQuantErrorSymmetricForSignedPtr = &QuantizeOptionsSetDumpQuantErrorSymmetricForSigned;
        mt->ShapeBucketOptionsSetEnablePtr = &ShapeBucketOptionsSetEnable;
        mt->ShapeBucketOptionsSetRangeInfoPtr = &ShapeBucketOptionsSetRangeInfo;
        mt->ShapeBucketOptionsSetSegmentsCountPtr = &ShapeBucketOptionsSetSegmentsCount;
        mt->ShapeBucketOptionsSetFixVarMapPtr = &ShapeBucketOptionsSetFixVarMap;
        mt->CpuTargetOptionsSetPackingPtr = &CpuTargetOptionsSetPacking;
        mt->RTValueFromHandlePtr = &RTValueFromHandle;
        mt->RTValueGetHandlePtr = &RTValueGetHandle;
        mt->StreamCreatePtr = &StreamCreate;
        mt->TargetCreatePtr = &TargetCreate;
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
            dataset.Chunk(dataset.Length / (int)samplesCount).Select(inputs => inputs.Zip(fnParams).ToDictionary(
            item => item.Second,
            item => item.First.ToValue()))).ToAsyncEnumerable();
        return GCHandle.ToIntPtr(GCHandle.Alloc(new CCalibrationDatasetProvider(samples, (int)samplesCount)));
    }

    [UnmanagedCallersOnly]
    private static void ClrHandleDispose(IntPtr handle)
    {
        Get<IDisposable>(handle).Dispose();
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
    private static DumpFlags CompileOptionsGetDumpFlags(IntPtr compileOptionsHandle)
    {
        return Get<CompileOptions>(compileOptionsHandle).DumpFlags;
    }

    [UnmanagedCallersOnly]
    private static void CompileOptionsSetDumpFlags(IntPtr compileOptionsHandle, DumpFlags dumpFlags)
    {
        Get<CompileOptions>(compileOptionsHandle).DumpFlags = dumpFlags;
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
    private static void CompileOptionsSetPreProcess(IntPtr compileOptionsHandle, byte preProcess)
    {
        switch (preProcess)
        {
            case 0:
                Get<CompileOptions>(compileOptionsHandle).PreProcess = false;
                break;
            case 1:
                Get<CompileOptions>(compileOptionsHandle).PreProcess = true;
                break;
            default:
                throw new ArgumentException("Invalid PreProcess Flag");
        }
    }

    [UnmanagedCallersOnly]
    private static void CompileOptionsSetInputLayout(IntPtr compileOptionsHandle, byte* inputLayout, nuint inputLayoutLength)
    {
        Get<CompileOptions>(compileOptionsHandle).InputLayout = ToString(inputLayout, inputLayoutLength);
    }

    [UnmanagedCallersOnly]
    private static void CompileOptionsSetOutputLayout(IntPtr compileOptionsHandle, byte* outputLayout, nuint outputLayoutLength)
    {
        Get<CompileOptions>(compileOptionsHandle).OutputLayout = ToString(outputLayout, outputLayoutLength);
    }

    [UnmanagedCallersOnly]
    private static void CompileOptionsSetInputType(IntPtr compileOptionsHandle, byte inputType)
    {
        // Get<CompileOptions>(compileOptionsHandle).InputType = inputType;
        switch (inputType)
        {
            case 0:
                Get<CompileOptions>(compileOptionsHandle).InputType = InputType.Uint8;
                break;
            case 1:
                Get<CompileOptions>(compileOptionsHandle).InputType = InputType.Int8;
                break;
            case 2:
                Get<CompileOptions>(compileOptionsHandle).InputType = InputType.Float32;
                break;
            default:
                throw new ArgumentException("Invalid InputType Flag");
        }
    }

    [UnmanagedCallersOnly]
    private static void CompileOptionsSetInputShape(IntPtr compileOptionsHandle, byte* inputShapeValue, nuint inputShapeValueLength)
    {
        Get<CompileOptions>(compileOptionsHandle).InputShape = StringToArrayInt32(ToString(inputShapeValue, inputShapeValueLength));
    }

    [UnmanagedCallersOnly]
    private static void CompileOptionsSetInputRange(IntPtr compileOptionsHandle, byte* inputRangeValue, nuint inputRangeValueLength)
    {
        Get<CompileOptions>(compileOptionsHandle).InputRange = StringToArrayFloat(ToString(inputRangeValue, inputRangeValueLength));
    }

    [UnmanagedCallersOnly]
    private static void CompileOptionsSetSwapRB(IntPtr compileOptionsHandle, byte swapRBValue)
    {
        switch (swapRBValue)
        {
            case 0:
                Get<CompileOptions>(compileOptionsHandle).SwapRB = false;
                break;
            case 1:
                Get<CompileOptions>(compileOptionsHandle).SwapRB = true;
                break;
            default:
                throw new ArgumentException("Invalid SwapRB Flag");
        }
    }

    [UnmanagedCallersOnly]
    private static void CompileOptionsSetLetterBoxValue(IntPtr compileOptionsHandle, float letterBoxValue)
    {
        Get<CompileOptions>(compileOptionsHandle).LetterBoxValue = letterBoxValue;
    }

    [UnmanagedCallersOnly]
    private static void CompileOptionsSetMean(IntPtr compileOptionsHandle, byte* meanValue, nuint meanValueLength)
    {
        Get<CompileOptions>(compileOptionsHandle).Mean = StringToArrayFloat(ToString(meanValue, meanValueLength));
    }

    [UnmanagedCallersOnly]
    private static void CompileOptionsSetStd(IntPtr compileOptionsHandle, byte* stdValue, nuint stdValueLength)
    {
        Get<CompileOptions>(compileOptionsHandle).Std = StringToArrayFloat(ToString(stdValue, stdValueLength));
    }

    [UnmanagedCallersOnly]
    private static void CompileOptionsSetShapeBucketOptions(IntPtr compileOptionsHandle, IntPtr shapeBucketOptionsHandle)
    {
        Get<CompileOptions>(compileOptionsHandle).ShapeBucketOptions = Get<ShapeBucketOptions>(shapeBucketOptionsHandle);
    }

    [UnmanagedCallersOnly]
    private static void CompileOptionsSetCpuTargetOptions(IntPtr compileOptionsHandle, IntPtr targetOptionsHandle)
    {
        Get<CompileOptions>(compileOptionsHandle).TargetOptions = Get<Targets.CpuTargetOptions>(targetOptionsHandle);
    }

    [UnmanagedCallersOnly]
    private static IntPtr CompileSessionCreate(IntPtr targetHandle, IntPtr compileOptionsHandle)
    {
        var target = Get<ITarget>(targetHandle);
        var compileOptions = Get<CompileOptions>(compileOptionsHandle);
        return GCHandle.ToIntPtr(GCHandle.Alloc(CompileSession.Create(target, compileOptions)));
    }

    [UnmanagedCallersOnly]
    private static IntPtr CompileSessionGetCompiler(IntPtr compileSessionHandle)
    {
        var compileSession = Get<CompileSession>(compileSessionHandle);
        return GCHandle.ToIntPtr(GCHandle.Alloc(compileSession.Compiler));
    }

    [UnmanagedCallersOnly]
    private static void CompilerInitialize()
    {
        var host = Host.CreateDefaultBuilder()
            .ConfigureCompiler()
            .Build();
        CompilerServices.Configure(host.Services);
    }

    [UnmanagedCallersOnly]
    private static IntPtr CompilerImportTFLiteModule(IntPtr compilerHandle, IntPtr streamHandle)
    {
        var compiler = Get<Compiler>(compilerHandle);
        var stream = Get<CStream>(streamHandle);
        var module = compiler.ImportTFLiteModuleAsync(stream).Result;
        return GCHandle.ToIntPtr(GCHandle.Alloc(module));
    }

    [UnmanagedCallersOnly]
    private static IntPtr CompilerImportOnnxModule(IntPtr compilerHandle, IntPtr streamHandle)
    {
        var compiler = Get<Compiler>(compilerHandle);
        var stream = Get<CStream>(streamHandle);
        var module = compiler.ImportOnnxModuleAsync(stream).Result;
        return GCHandle.ToIntPtr(GCHandle.Alloc(module));
    }

    [UnmanagedCallersOnly]
    private static IntPtr CompilerImportNcnnModule(IntPtr compilerHandle, IntPtr ncnnParamHandle, IntPtr ncnnBinHandle)
    {
        var compiler = Get<Compiler>(compilerHandle);
        var ncnnParam = Get<CStream>(ncnnParamHandle);
        var ncnnBin = Get<CStream>(ncnnBinHandle);
        var module = compiler.ImportNcnnModuleAsync(ncnnParam, ncnnBin).Result;
        return GCHandle.ToIntPtr(GCHandle.Alloc(module));
    }

    [UnmanagedCallersOnly]
    private static void CompilerCompile(IntPtr compilerHandle)
    {
        var compiler = Get<Compiler>(compilerHandle);
        compiler.CompileAsync().Wait();
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
            x => x.First,
            x => x.Second.ToValue()));
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
    private static IntPtr ShapeBucketOptionsCreate()
    {
        return GCHandle.ToIntPtr(GCHandle.Alloc(ShapeBucketOptions.Default));
    }

    [UnmanagedCallersOnly]
    private static IntPtr CpuTargetOptionsCreate()
    {
        return GCHandle.ToIntPtr(GCHandle.Alloc(new Targets.CpuTargetOptions()));
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
    private static void QuantizeOptionsSetModelQuantMode(IntPtr quantizeOptionsHandle, ModelQuantMode quantMode)
    {
        Get<QuantizeOptions>(quantizeOptionsHandle).ModelQuantMode = quantMode;
    }

    [UnmanagedCallersOnly]
    private static void QuantizeOptionsSetQuantType(IntPtr quantizeOptionsHandle, QuantType quantType)
    {
        switch (quantType)
        {
            case QuantType.Uint8:
                Get<QuantizeOptions>(quantizeOptionsHandle).QuantType = DataTypes.UInt8;
                break;
            case QuantType.Int8:
                Get<QuantizeOptions>(quantizeOptionsHandle).QuantType = DataTypes.Int8;
                break;
            case QuantType.Int16:
                Get<QuantizeOptions>(quantizeOptionsHandle).QuantType = DataTypes.Int16;
                break;
            default:
                throw new ArgumentException("Not Supported Quant Type");
        }
    }

    [UnmanagedCallersOnly]
    private static void QuantizeOptionsSetWQuantType(IntPtr quantizeOptionsHandle, QuantType wQuantType)
    {
        switch (wQuantType)
        {
            case QuantType.Uint8:
                Get<QuantizeOptions>(quantizeOptionsHandle).WQuantType = DataTypes.UInt8;
                break;
            case QuantType.Int8:
                Get<QuantizeOptions>(quantizeOptionsHandle).WQuantType = DataTypes.Int8;
                break;
            case QuantType.Int16:
                Get<QuantizeOptions>(quantizeOptionsHandle).WQuantType = DataTypes.Int16;
                break;
            default:
                throw new ArgumentException("Not Supported Weights Quant Type");
        }
    }

    [UnmanagedCallersOnly]
    private static void QuantizeOptionsSetFineTuneWeightsMethod(IntPtr quantizeOptionsHandle, FineTuneWeightsMethod fineTuneWeightsMethod)
    {
        switch (fineTuneWeightsMethod)
        {
            case FineTuneWeightsMethod.NoFineTuneWeights:
                Get<QuantizeOptions>(quantizeOptionsHandle).UseSquant = false;
                Get<QuantizeOptions>(quantizeOptionsHandle).UseAdaRound = false;
                break;
            case FineTuneWeightsMethod.UseSquant:
                Get<QuantizeOptions>(quantizeOptionsHandle).UseSquant = true;
                Get<QuantizeOptions>(quantizeOptionsHandle).UseAdaRound = false;
                break;
            case FineTuneWeightsMethod.UseAdaRound:
                Get<QuantizeOptions>(quantizeOptionsHandle).UseSquant = false;
                Get<QuantizeOptions>(quantizeOptionsHandle).UseAdaRound = true;
                break;
            default:
                throw new ArgumentException("Not Supported Finetune Weights Method");
        }
    }

    [UnmanagedCallersOnly]
    private static void QuantOptionsSetUseMixQuant(IntPtr quantizeOptionsHandle, byte useMixQuant)
    {
        switch (useMixQuant)
        {
            case 0:
                Get<QuantizeOptions>(quantizeOptionsHandle).BindQuantMethod = false;
                break;
            case 1:
                Get<QuantizeOptions>(quantizeOptionsHandle).BindQuantMethod = true;
                break;
            default:
                throw new ArgumentException("Invalid useMixQuant Flag");
        }
    }

    [UnmanagedCallersOnly]
    private static void QuantizeOptionsSetQuantScheme(IntPtr quantizeOptionsHandle, byte* quantSchemePtr, nuint quantSchemeLength)
    {
        Get<QuantizeOptions>(quantizeOptionsHandle).QuantScheme = ToString(quantSchemePtr, quantSchemeLength);
    }

    [UnmanagedCallersOnly]
    private static void QuantizeOptionsSetQuantSchemeStrictMode(IntPtr quantizeOptionsHandle, byte quantSchemeStrictMode)
    {
        switch (quantSchemeStrictMode)
        {
            case 0:
                Get<QuantizeOptions>(quantizeOptionsHandle).QuantSchemeStrictMode = false;
                break;
            case 1:
                Get<QuantizeOptions>(quantizeOptionsHandle).QuantSchemeStrictMode = true;
                break;
            default:
                throw new ArgumentException("Invalid QuantSchemeStrictMode Flag");
        }
    }

    [UnmanagedCallersOnly]
    private static void QuantizeOptionsSetExportQuantScheme(IntPtr quantizeOptionsHandle, byte exportQuantScheme)
    {
        switch (exportQuantScheme)
        {
            case 0:
                Get<QuantizeOptions>(quantizeOptionsHandle).ExportQuantScheme = false;
                break;
            case 1:
                Get<QuantizeOptions>(quantizeOptionsHandle).ExportQuantScheme = true;
                break;
            default:
                throw new ArgumentException("Invalid exportQuantScheme Flag");
        }
    }

    [UnmanagedCallersOnly]
    private static void QuantizeOptionsSetExportWeightRangeByChannel(IntPtr quantizeOptionsHandle, byte exportWeightRangeByChannel)
    {
        switch (exportWeightRangeByChannel)
        {
            case 0:
                Get<QuantizeOptions>(quantizeOptionsHandle).ExportWeightRangeByChannel = false;
                break;
            case 1:
                Get<QuantizeOptions>(quantizeOptionsHandle).ExportWeightRangeByChannel = true;
                break;
            default:
                throw new ArgumentException("Invalid exportWeightRangeByChannel Flag");
        }
    }

    [UnmanagedCallersOnly]
    private static void QuantizeOptionsSetDumpQuantError(IntPtr quantizeOptionsHandle, byte dumpQuantError)
    {
        switch (dumpQuantError)
        {
            case 0:
                Get<QuantizeOptions>(quantizeOptionsHandle).DumpQuantError = false;
                break;
            case 1:
                Get<QuantizeOptions>(quantizeOptionsHandle).DumpQuantError = true;
                break;
            default:
                throw new ArgumentException("Invalid dumpQuantError Flag");
        }
    }

    [UnmanagedCallersOnly]
    private static void QuantizeOptionsSetDumpQuantErrorSymmetricForSigned(IntPtr quantizeOptionsHandle, byte dumpQuantErrorSymmetricForSigned)
    {
        switch (dumpQuantErrorSymmetricForSigned)
        {
            case 0:
                Get<QuantizeOptions>(quantizeOptionsHandle).DumpQuantErrorSymmetricForSigned = false;
                break;
            case 1:
                Get<QuantizeOptions>(quantizeOptionsHandle).DumpQuantErrorSymmetricForSigned = true;
                break;
            default:
                throw new ArgumentException("Invalid dumpQuantErrorSymmetricForSigned Flag");
        }
    }

    [UnmanagedCallersOnly]
    private static void ShapeBucketOptionsSetEnable(IntPtr shapeBucketOptionsHandle, byte enable)
    {
        switch (enable)
        {
            case 0:
                Get<ShapeBucketOptions>(shapeBucketOptionsHandle).Enable = false;
                break;
            case 1:
                Get<ShapeBucketOptions>(shapeBucketOptionsHandle).Enable = true;
                break;
            default:
                throw new ArgumentException("Invalid enable Flag");
        }
    }

    [UnmanagedCallersOnly]
    private static void ShapeBucketOptionsSetRangeInfo(IntPtr shapeBucketOptionsHandle, byte* rangeInfo, nuint rangeInfoSize)
    {
        byte[] rangeInfoByte = new byte[rangeInfoSize];
        Marshal.Copy(new IntPtr(rangeInfo), rangeInfoByte, 0, (int)rangeInfoSize);
        string jsonStr = Encoding.UTF8.GetString(rangeInfoByte);
        Dictionary<string, List<int>> rangeInfoStructTmp = JsonConvert.DeserializeObject<Dictionary<string, List<int>>>(jsonStr)!;
        Dictionary<string, (int, int)> rangeInfoStruct = new();
        foreach (var element in rangeInfoStructTmp)
        {
            rangeInfoStruct.Add(element.Key, (element.Value[0], element.Value[1]));
        }

        Get<ShapeBucketOptions>(shapeBucketOptionsHandle).RangeInfo = rangeInfoStruct;
    }

    [UnmanagedCallersOnly]
    private static void ShapeBucketOptionsSetSegmentsCount(IntPtr shapeBucketOptionsHandle, nuint segmentCount)
    {
        Get<ShapeBucketOptions>(shapeBucketOptionsHandle).SegmentsCount = (int)segmentCount;
    }

    [UnmanagedCallersOnly]
    private static void ShapeBucketOptionsSetFixVarMap(IntPtr shapeBucketOptionsHandle, byte* fixVarMap, nuint fixVarMapSize)
    {
        byte[] fixVarMapByte = new byte[fixVarMapSize];
        Marshal.Copy(new IntPtr(fixVarMap), fixVarMapByte, 0, (int)fixVarMapSize);
        string jsonStr = Encoding.UTF8.GetString(fixVarMapByte);
        Dictionary<string, int> fixVarMapStruct = JsonConvert.DeserializeObject<Dictionary<string, int>>(jsonStr)!;
        Get<ShapeBucketOptions>(shapeBucketOptionsHandle).FixVarMap = fixVarMapStruct;
    }

    [UnmanagedCallersOnly]
    private static void CpuTargetOptionsSetPacking(IntPtr handle, byte packing)
    {
        Get<Targets.CpuTargetOptions>(handle).Packing = packing != 0;
    }

    [UnmanagedCallersOnly]
    private static IntPtr RTValueFromHandle(IntPtr handle)
    {
        var rtValue = RTValue.FromHandle(handle, true);
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
    private static IntPtr TargetCreate(byte* targetNamePtr, nuint targetNameLength)
    {
        var targetName = ToString(targetNamePtr, targetNameLength);
        return GCHandle.ToIntPtr(GCHandle.Alloc(CompilerServices.GetTarget(targetName)));
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

    private static T[] To1DArray<T>(T* value, nuint shape0)
        where T : unmanaged
    {
        var arr = new T[shape0];
        for (nuint i = 0; i < shape0; i++)
        {
            arr[i] = value[i];
        }

        return arr;
    }

    private static T[][] To2DArray<T>(T* value, nuint* shape1, nuint shape0)
        where T : unmanaged
    {
        var arr = new T[shape0][];
        int count = 0;
        for (nuint i = 0; i < shape0; i++)
        {
            arr[i] = new T[shape1[i]];
            for (nuint j = 0; j < shape1[i]; j++)
            {
                arr[i][j] = value[count++];
            }
        }

        return arr;
    }

    private static int[] StringToArrayInt32(string value)
    {
        var data = value.Replace(" ", string.Empty, StringComparison.OrdinalIgnoreCase).Split(",");
        return Array.ConvertAll(data, int.Parse);
    }

    private static float[] StringToArrayFloat(string value)
    {
        var data = value.Replace(" ", string.Empty, StringComparison.OrdinalIgnoreCase).Split(',');
        return Array.ConvertAll(data, float.Parse);
    }

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
