// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text.RegularExpressions;

namespace CApiGen;

public enum NumKind
{
    Integer,
    Floating,
    None,
}

public static class CApiGenExtensions
{
    private static readonly Dictionary<TypeCode, string> BuiltInMemo =
    new Dictionary<TypeCode, string>()
    {
        { TypeCode.Boolean, "bool" },
        { TypeCode.Char, "char" },
        { TypeCode.SByte, "sbyte" },
        { TypeCode.Byte, "byte" },
        { TypeCode.Int16, "int16" },
        { TypeCode.UInt16, "uint16" },
        { TypeCode.Int32, "int" },
        { TypeCode.UInt32, "uint" },
        { TypeCode.Int64, "long" },
        { TypeCode.UInt64, "ulong" },
        { TypeCode.Single, "float" },
        { TypeCode.Double, "double" },
        { TypeCode.String, "string" },
    };

    private static readonly Dictionary<string, string> UnCPPMemo = new()
    {
        { "byte", "uint8_t" },
        { "sbyte", "int8_t" },
        { "short", "int16_t" },
        { "ushort", "uint16_t" },
        { "int", "int32_t" },
        { "uint", "uint32_t" },
        { "long", "int64_t" },
        { "ulong", "uint64_t" },
        { "nint", "intptr_t" },
        { "nuint", "uintptr_t" },
        { "float", "float32_t" },
        { "double", "float64_t" },
        { "char", "char" },
        { "string", "const char*" },
    };

    public static string ToSnake(this string input)
    {
        // Use regex to insert underscores before capital letters
        string snakeCase = Regex.Replace(input, "([a-z])([A-Z])", "$1_$2").ToLower(System.Globalization.CultureInfo.CurrentCulture);
        return snakeCase;
    }

    public static NumKind IsNumeric(this Type type)
    {
        if (type.GetInterfaces().Any(i => i.IsGenericType && (i.GetGenericTypeDefinition() == typeof(IFloatingPoint<>))))
        {
            return NumKind.Floating;
        }
        else if (type.GetInterfaces().Any(i => i.IsGenericType && (i.GetGenericTypeDefinition() == typeof(INumber<>))))
        {
            return NumKind.Integer;
        }

        return NumKind.None;
    }

    public static string RenderValue(this object? value, LangMode langMode)
    {
        return value switch
        {
            bool b => b.ToString().ToLower(System.Globalization.CultureInfo.CurrentCulture),
            Enum e => $"{e.GetType().Name}.{e}",
            null => throw new NotSupportedException("null"),
            _ => value.ToString()!,
        };
    }

    public static IEnumerable<(string Name, string Value)> RenderEnumFields(this Type type, LangMode langMode)
    {
        var names = type.GetEnumNames();
        var values = type.GetEnumValuesAsUnderlyingType();
        var outValues = new List<string>();
        for (int i = 0; i < names.Length; i++)
        {
            outValues.Add(RenderValue(values.GetValue(i), langMode));
        }

        return names.Zip(outValues);
    }

    public static string RenderTypeCode(Type type, TypeCode code, string alias, LangMode langMode)
    {
        return langMode switch
        {
            LangMode.CSC => alias,
            LangMode.UnCS => alias switch
            {
                "string" => "byte*",
                "bool" => "byte",
                _ => alias,
            },
            LangMode.UnCPP => code switch
            {
                TypeCode.Boolean => UnCPPMemo["byte"],
                _ => UnCPPMemo[alias],
            },
            LangMode.Pyb => code switch
            {
                TypeCode.Boolean => "bool",
                TypeCode.String => "std::string_view",
                TypeCode.Int16 or TypeCode.Int32 or TypeCode.Int64 => "int",
                TypeCode.Single or TypeCode.Double => "float",
                _ => throw new NotSupportedException(),
            },
            LangMode.Pyi => code switch
            {
                TypeCode.String => "str",
                TypeCode.Boolean => "bool",
                _ => IsNumeric(type) switch
                {
                    NumKind.Integer => "int",
                    NumKind.Floating => "float",
                    _ => throw new NotSupportedException(),
                },
            },
            _ => throw new System.Diagnostics.UnreachableException(nameof(langMode)),
        };
    }

    public static string RenderType(this Type type, LangMode langMode = LangMode.CSC)
    {
        var code = Type.GetTypeCode(type);
        if (type.IsEnum)
        {
            return langMode switch
            {
                LangMode.CSC or LangMode.UnCS => type.Name,
                LangMode.UnCPP => RenderTypeCode(type, code, BuiltInMemo[code], langMode),
                LangMode.Pyb => $"{type.Name.ToSnake()}_t",
                LangMode.Pyi => type.Name,
                _ => throw new System.Diagnostics.UnreachableException(nameof(langMode)),
            };
        }
        else if (BuiltInMemo.TryGetValue(code, out var alias))
        {
            return RenderTypeCode(type, code, alias, langMode);
        }
        else if (type.IsArray)
        {
            if (type.IsNestedArrayType(out var stacks))
            {
                switch (langMode)
                {
                    case LangMode.CSC:
                        {
                            var s = RenderType(stacks.Last(), langMode);
                            for (int i = stacks.Count - 2; i > 0; i--)
                            {
                                s = $"{s}[]";
                            }

                            return $"IEnumerable<{s}>";
                        }

                    case LangMode.UnCPP:
                        return RenderType(stacks.Last()) + "*";
                    case LangMode.Pyi:
                        {
                            var s = RenderType(stacks.Last(), langMode);
                            for (int i = stacks.Count - 2; i >= 0; i--)
                            {
                                s = $"List[{s}]";
                            }

                            return s;
                        }

                    case LangMode.Pyb:
                        {
                            var s = RenderType(stacks.Last(), langMode);
                            for (int i = stacks.Count - 2; i >= 0; i--)
                            {
                                s = $"std::vector<{s}>";
                            }

                            return s;
                        }

                    default:
                        throw new NotSupportedException(langMode.ToString());
                }
            }
            else
            {
                throw new NotSupportedException("multi dimension array");
            }
        }
        else
        {
            throw new NotSupportedException(type.FullName);
        }
    }

    /// <summary>
    /// check type is nested array type.
    /// </summary>
    /// <param name="type">type.</param>
    /// <param name="stacks">nest stack, contains itself. </param>
    /// <returns> is nested array. </returns>
    public static bool IsNestedArrayType(this Type type, out List<Type> stacks)
    {
        stacks = new();
        if (type.IsArray && type.GetArrayRank() == 1 && type.GetElementType() is Type inner)
        {
            stacks.Add(type);
            if (IsNestedArrayType(inner, out var childInners))
            {
                stacks.AddRange(childInners);
            }
            else
            {
                stacks.Add(inner);
            }

            return true;
        }

        return false;
    }
}
