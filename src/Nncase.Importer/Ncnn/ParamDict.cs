// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Google.Protobuf.WellKnownTypes;
using Nncase.IR;

namespace Nncase.Importer.Ncnn;

internal enum ParamKind
{
    Null,
    IntOrFloat,
    Int,
    Float,
    ArrayOfIntOrFloat,
    ArrayOfInt,
    ArrayOfFloat,
}

internal struct ParamValue
{
    public ParamKind Kind;

    public int IntValue;

    public float FloatValue;

    public Tensor? TensorValue;
}

internal class ParamDict
{
    public static readonly int NcnnMaxParamCount = 32;

    public void LoadFrom(ReadOnlySpan<string> fields)
    {
        foreach (var field in fields)
        {
            if (field.Split('=', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries) is not [var idStr, var valueStr])
            {
                break;
            }

            var id = int.Parse(idStr);
            if (id >= NcnnMaxParamCount)
            {
                throw new InvalidDataException($"id < NCNN_MAX_PARAM_COUNT failed (id={id}, NCNN_MAX_PARAM_COUNT={NcnnMaxParamCount})");
            }

            var paramValue = default(ParamValue);
            var isArray = id <= -23300;
            var isFloat = valueStr.AsSpan().IndexOfAny('.', 'e', 'E') != -1;
            if (isArray)
            {
                var elements = valueStr.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
                if (elements == null || elements.Length == 0)
                {
                    throw new InvalidDataException("ParamDict read array length failed");
                }

                var length = int.Parse(elements[0]);
                if (isFloat)
                {
                    var value = new Tensor<float>(length);
                    for (var i = 1; i < length; i++)
                    {
                        value[i] = float.Parse(elements[i + 1]);
                    }

                    paramValue.Kind = ParamKind.ArrayOfFloat;
                    paramValue.TensorValue = value;
                }
                else
                {
                    var value = new Tensor<int>(length);
                    for (var i = 1; i < length; i++)
                    {
                        value[i] = int.Parse(elements[i + 1]);
                    }

                    paramValue.Kind = ParamKind.ArrayOfInt;
                    paramValue.TensorValue = value;
                }
            }
            else
            {
                if (isFloat)
                {
                    paramValue.Kind = ParamKind.Float;
                    paramValue.FloatValue = float.Parse(valueStr);
                }
                else
                {
                    paramValue.Kind = ParamKind.Int;
                    paramValue.IntValue = int.Parse(valueStr);
                }
            }
        }
    }
}
