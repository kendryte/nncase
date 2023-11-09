// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reactive;
using System.Reactive.Linq;
using System.Threading;
using System.Threading.Tasks;
using Avalonia.Platform.Storage;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Nncase.IR;
using Nncase.Quantization;
using Nncase.Studio.Views;
using NumSharp;
using ReactiveUI;

namespace Nncase.Studio.ViewModels;

public static class DataUtil
{
    public static string TensorTypeToString(TensorType tt)
    {
        return $"{tt.DType} {tt.Shape}";
    }

    public static string VarToString(Var var)
    {
        var tt = (TensorType)var.TypeAnnotation;
        return $"{var.Name} {TensorTypeToString(tt)}";
    }

    public static (string[] InputFiles, Tensor[] InputList) ReadMultiInputs(List<string> path)
    {
        var inputFiles = path.Count == 1 && Directory.Exists(path[0])
            ? Directory.GetFiles(path[0])
            : path.ToArray();
        var input = ReadInput(inputFiles).ToArray();
        return (inputFiles, input);
    }

    public static List<Tensor> ReadInput(string[] file)
    {
        return file
            .Where(f => Path.GetExtension(f) == ".npy")
            .Select(f =>
            {
                var tensor = np.load(f);
                return Tensor.FromBytes(new TensorType(ToDataType(tensor.dtype), tensor.shape), tensor.ToByteArray());
            }).ToList();
    }

    public static DataType ToDataType(Type type)
    {
        if (type == typeof(byte))
        {
            return DataTypes.UInt8;
        }

        if (type == typeof(sbyte))
        {
            return DataTypes.Int8;
        }

        if (type == typeof(ushort))
        {
            return DataTypes.UInt16;
        }

        if (type == typeof(short))
        {
            return DataTypes.Int16;
        }

        if (type == typeof(int))
        {
            return DataTypes.Int32;
        }

        if (type == typeof(uint))
        {
            return DataTypes.UInt32;
        }

        if (type == typeof(long))
        {
            return DataTypes.Int64;
        }

        if (type == typeof(ulong))
        {
            return DataTypes.UInt64;
        }

        if (type == typeof(float))
        {
            return DataTypes.Float32;
        }

        if (type == typeof(double))
        {
            return DataTypes.Float64;
        }

        if (type == typeof(bool))
        {
            return DataTypes.Boolean;
        }

        if (type == typeof(Half))
        {
            return DataTypes.Float16;
        }

        throw new NotImplementedException($"not supported data type {type}");
    }

    public static DataType QuantTypeToDataType(QuantType qt)
    {
        return qt switch
        {
            QuantType.Uint8 => DataTypes.UInt8,
            QuantType.Int8 => DataTypes.Int8,
            QuantType.Int16 => DataTypes.Int16,
            _ => throw new ArgumentOutOfRangeException(nameof(qt), qt, null),
        };
    }
}
