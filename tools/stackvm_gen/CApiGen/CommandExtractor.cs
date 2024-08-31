// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.CodeDom;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using Microsoft.CSharp;
using Nncase.CommandLine;
using RazorLight;
using RazorLight.Razor;

namespace CApiGen;

[Flags]
public enum SignMode : byte
{
    Type,
    Param,
}

public enum LangMode : byte
{
    /* csharp command. */
    CSC,

    /* unmanged csharp */
    UnCS,

    /* unmanged cpp */
    UnCPP,

    /* pybind */
    Pyb,

    /* pyi */
    Pyi,
}

public class CommandExtractor
{
    private readonly RazorLightEngine _engine;

    public CommandExtractor()
    {
        _engine = new RazorLightEngineBuilder()
            .UseMemoryCachingProvider()
            .UseProject(new EmbeddedRazorProject(typeof(Program)) { Extension = ".razor" })
            .Build();

        OptionInfos = new();
        OptionsType = typeof(int);
    }

    public List<OptionInfo> OptionInfos { get; }

    public Type OptionsType { get; set; }

    public void Extract(Type options)
    {
        if (!options.GetInterfaces().Contains(typeof(Nncase.ITargetOptions)))
        {
            throw new NotSupportedException();
        }

        OptionsType = options;

        foreach (var info in options.GetProperties())
        {
            var option = new OptionInfo(
                info.PropertyType,
                info.Name,
                info.GetCustomAttribute<DisplayNameAttribute>()?.DisplayName!,
                info.GetCustomAttribute<DescriptionAttribute>()?.Description!,
                info.GetCustomAttribute<DefaultValueAttribute>()?.Value!,
                info.GetCustomAttribute<FromAmongAttribute>()?.Values ?? Array.Empty<object>(),
                info.GetCustomAttribute<AmbientValueAttribute>()?.Value is string s ? s : null,
                info.GetCustomAttribute<AllowMultiplePerTokenAttribute>() is null ? false : true);
            OptionInfos.Add(option);
        }
    }

    public Task<string> RenderAsync(string templateName)
    {
        return _engine.CompileRenderAsync(templateName, this);
    }
}

public record class OptionInfo(Type PropertyType, string PropertyName, string DisplayName, string Description, object DefulatValue, object[] Amongs, string? Parser, bool AllowMultiple = false)
{
    /// <summary>
    /// render csharp command get default value.
    /// </summary>
    /// <returns>str.</returns>
    public string RenderCSCDefaultValue()
    {
        return DefulatValue switch
        {
            string v when v.StartsWith("() => ") => v,
            string v when v == string.Empty => $"() => string.Empty",
            string v => $"() => \"{v}\"",
            _ => $"() => {DefulatValue.RenderValue(LangMode.CSC)}",
        };
    }

    public string RenderSignature(SignMode signMode, LangMode langMode)
    {
        var sizeType = langMode switch
        {
            LangMode.UnCS => "nuint",
            LangMode.UnCPP => "size_t",
            _ => throw new NotSupportedException(nameof(langMode)),
        };

        var sb = new StringBuilder();
        if (PropertyType.IsValueType)
        {
            if (signMode.HasFlag(SignMode.Type))
            {
                sb.Append((PropertyType == typeof(bool) ? typeof(byte) : PropertyType).RenderType(langMode));
            }

            if (signMode.HasFlag(SignMode.Param))
            {
                if (signMode.HasFlag(SignMode.Type))
                {
                    sb.Append(' ');
                }

                sb.Append("value");
            }
        }
        else if (PropertyType.IsArray && PropertyType.GetElementType() is Type elemType)
        {
            var typeList = new List<Type> { elemType };
            while (elemType.IsArray && elemType.GetElementType() is Type inner)
            {
                typeList.Add(inner);
                elemType = inner;
            }

            var elemTypeName = elemType.RenderType(langMode);
            if (signMode.HasFlag(SignMode.Type))
            {
                sb.Append($"{elemTypeName}*");
            }

            if (signMode.HasFlag(SignMode.Param))
            {
                if (signMode.HasFlag(SignMode.Type))
                {
                    sb.Append(' ');
                }

                sb.Append("value");
            }

            for (int i = typeList.Count; i > 0; i--)
            {
                var suffix = i == 1 ? string.Empty : "*";
                if (signMode.HasFlag(SignMode.Type))
                {
                    sb.Append($", {sizeType}{suffix}");
                }

                if (signMode.HasFlag(SignMode.Param))
                {
                    if (signMode.HasFlag(SignMode.Type))
                    {
                        sb.Append(' ');
                    }
                    else
                    {
                        sb.Append(", ");
                    }

                    sb.Append($"shape{i - 1}");
                }
            }
        }
        else if (PropertyType == typeof(string))
        {
            if (signMode.HasFlag(SignMode.Type))
            {
                sb.Append(PropertyType.RenderType(langMode));
            }

            if (signMode.HasFlag(SignMode.Param))
            {
                if (signMode.HasFlag(SignMode.Type))
                {
                    sb.Append(' ');
                }

                sb.Append("value");
            }

            if (signMode.HasFlag(SignMode.Type))
            {
                sb.Append($", {sizeType}");
            }

            if (signMode.HasFlag(SignMode.Param))
            {
                if (signMode.HasFlag(SignMode.Type))
                {
                    sb.Append(' ');
                }
                else
                {
                    sb.Append(", ");
                }

                sb.Append("length");
            }

            return sb.ToString();
        }
        else
        {
            throw new NotSupportedException();
        }

        return sb.ToString();
    }

    public string RenderUnCSAssginValue()
    {
        var sb = new StringBuilder();
        if (PropertyType.IsValueType)
        {
            sb.Append(PropertyType == typeof(bool) ? "value != 0" : "value");
        }
        else if (PropertyType.IsArray)
        {
            if (PropertyType.IsNestedArrayType(out var inners))
            {
                var rank = inners.Count;
                sb.Append($"To{rank}DArray(value");
                for (int i = rank; i > 0; i--)
                {
                    sb.Append($", shape{i - 1}");
                }

                sb.Append(')');
            }
            else
            {
                throw new NotSupportedException();
            }
        }
        else if (PropertyType == typeof(string))
        {
            sb.Append($"ToString(value, length)");
        }
        else
        {
            throw new NotSupportedException();
        }

        return sb.ToString();
    }
}
