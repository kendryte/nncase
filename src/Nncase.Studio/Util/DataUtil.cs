﻿// Copyright (c) Canaan Inc. All rights reserved.
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
using DynamicData;
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

    public static Tensor ReadNumpyAsTensor(string f)
    {
        var tensor = np.load(f);
        return Tensor.FromBytes(new TensorType(DataType.FromType(tensor.dtype), tensor.shape), tensor.ToByteArray());
    }

    public static List<Tensor> ReadInput(string[] file)
    {
        return file
            .Where(f => Path.GetExtension(f) == ".npy")
            .Select(ReadNumpyAsTensor).ToList();
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

    public static bool TryParseFixVarMap(string input, out Dictionary<string, int> map)
    {
        map = new();
        if (input == string.Empty)
        {
            return false;
        }

        try
        {
            if (!input.Contains(":", StringComparison.Ordinal))
            {
                return false;
            }

            map = input.Trim().Split(",").Select(x => x.Trim().Split(":")).ToDictionary(x => x[0], x => int.Parse(x[1]));
            return true;
        }
        catch (Exception)
        {
            return false;
        }
    }

    public static bool TryParseRangeInfo(string input, out Dictionary<string, (int Min, int Max)> map)
    {
        map = new();
        if (input == string.Empty)
        {
            return false;
        }

        try
        {
            map = input.Trim()
                .Split(";")
                .Select(x => x.Trim().Split(":"))
                .ToDictionary(x => x[0], x =>
                {
                    var pair = x[1].Split(",");
                    return (int.Parse(pair[0].Trim('(')), int.Parse(pair[1].Trim(')')));
                });
            return true;
        }
        catch (Exception)
        {
            return false;
        }
    }
}
