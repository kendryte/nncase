// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.Runtime.Ncnn;

public enum ParamKind
{
    Null,
    IntOrFloat,
    Int,
    Float,
    ArrayOfIntOrFloat,
    ArrayOfInt,
    ArrayOfFloat,
}

public struct ParamValue
{
    public ParamKind Kind;

    public int IntValue;

    public float FloatValue;

    public Tensor? TensorValue;
}

public class ParamDict
{
    public static readonly int NcnnMaxParamCount = 32;

    private readonly Dictionary<int, ParamValue> _values = new();

    public ParamValue this[int index]
    {
        get => _values[index];
        set => _values[index] = value;
    }

    public void LoadFrom(ReadOnlySpan<string> fields)
    {
        foreach (var field in fields)
        {
            if (field.Split('=', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries) is not[var idStr, var valueStr])
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
                id = -id - 23300;
                var elements = valueStr.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
                if (elements == null || elements.Length == 0)
                {
                    throw new InvalidDataException("ParamDict read array length failed");
                }

                var length = int.Parse(elements[0]);
                if (isFloat)
                {
                    var value = new Tensor<float>(length);
                    for (var i = 0; i < length; i++)
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

            _values.Add(id, paramValue);
        }
    }

    public int Get(int id, int defaultValue) => _values.TryGetValue(id, out var value) ? value.IntValue : defaultValue;

    public float Get(int id, float defaultValue) => _values.TryGetValue(id, out var value) ? value.FloatValue : defaultValue;

    public Tensor<T> Get<T>(int id, Tensor<T> defaultValue)
        where T : unmanaged, IEquatable<T>
        => _values.TryGetValue(id, out var value) ? value.TensorValue!.Cast<T>() : defaultValue;

    public void Serialize(TextWriter writer)
    {
        int index = 0;
        foreach (var field in _values)
        {
            var id = field.Key;

            var paramValue = field.Value;
            var isArray = paramValue.Kind is ParamKind.ArrayOfFloat or ParamKind.ArrayOfIntOrFloat or ParamKind.ArrayOfInt;
            var isFloat = paramValue.Kind is ParamKind.ArrayOfFloat or ParamKind.Float;

            if (isArray)
            {
                id = id - 23300;
            }

            writer.Write($"{id}=");

            if (isArray)
            {
                if (isFloat)
                {
                    writer.Write(string.Join(',', paramValue.TensorValue!.Cast<float>().Select(x => x.ToString("e"))));
                }
                else
                {
                    writer.Write(string.Join(',', paramValue.TensorValue!.Cast<int>()));
                }
            }
            else
            {
                if (isFloat)
                {
                    writer.Write(paramValue.FloatValue.ToString("e"));
                }
                else
                {
                    writer.Write(paramValue.IntValue);
                }
            }

            if (++index != _values.Count)
            {
                writer.Write(' ');
            }
        }
    }
}
