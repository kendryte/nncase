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
    Type = 1 << 1,
    Param = 1 << 2,
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

    public string RenderPybAssginNestArray(Type type, List<Type> stacks, int ind)
    {
        var sb = new StringBuilder();
        var indent = string.Join(string.Empty, Enumerable.Repeat(' ', ind));
        sb.AppendLine($"{indent}std::vector<int> values;");
        var st = "size_t";
        for (int i = 0; i < stacks.Count - 1; i++)
        {
            sb.AppendLine($"{indent}{st} shape{i};");
            st = $"std::vector<{st}>";
        }

        string SubScript(int level)
        {
            return level switch
            {
                0 => string.Empty,
                1 => "[i0]",
                2 => "[i0][i1]",
                _ => throw new NotSupportedException(),
            };
        }

        void RenderLoop(int l, int cind)
        {
            var cindent = string.Join(string.Empty, Enumerable.Repeat(' ', cind));
            if (l < stacks.Count - 1)
            {
                var size = $"value{SubScript(l)}.size()";
                var assign = l == 0 ? $" = {size}" : $".push_back({size})";
                sb.AppendLine($"{cindent}shape{l}{assign};");
                sb.AppendLine($"{cindent}for (size_t i{l} = 0; i{l} < shape{l}{SubScript(l)}; i{l}++) {{");
                RenderLoop(l + 1, cind += 2);
                sb.AppendLine($"{cindent}}}");
            }
            else
            {
                sb.AppendLine($"{cindent}values.push_back(value{SubScript(l)});");
            }
        }

        RenderLoop(0, ind);
        return sb.ToString();
    }

    public string RenderSignature(SignMode signMode, LangMode langMode)
    {
        var sizeType = langMode switch
        {
            LangMode.UnCS => "nuint",
            LangMode.UnCPP or LangMode.Pyb => "size_t",
            _ => throw new NotSupportedException(nameof(langMode)),
        };

        var sb = new StringBuilder();
        if (PropertyType.IsValueType)
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
        }
        else if (PropertyType.IsArray)
        {
            if (PropertyType.IsNestedArrayType(out var stacks))
            {
                // 1. process array type.
                var elemTypeName = stacks.Last().RenderType(langMode);
                if (signMode.HasFlag(SignMode.Type))
                {
                    if (langMode is LangMode.Pyb)
                    {
                        sb.Append(PropertyType.RenderType(langMode));
                    }
                    else
                    {
                        sb.Append($"{elemTypeName}*");
                    }
                }

                if (signMode.HasFlag(SignMode.Param))
                {
                    if (signMode.HasFlag(SignMode.Type))
                    {
                        sb.Append(' ');
                    }

                    if (langMode is LangMode.Pyb && signMode == SignMode.Param)
                    {
                        sb.Append("values.data()");
                    }
                    else
                    {
                        sb.Append("value");
                    }
                }

                // 2. process nested.
                for (int i = 0; i < stacks.Count - 1; i++)
                {
                    // the pybind function param no need nested.
                    if (langMode is LangMode.Pyb && signMode == (SignMode.Type | SignMode.Param))
                    {
                        break;
                    }

                    var suffix = string.Join(string.Empty, Enumerable.Repeat('*', i));
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

                        if (langMode is LangMode.Pyb && signMode == SignMode.Param)
                        {
                            var sf = i > 0 ? ".data()" : string.Empty;
                            sb.Append($"shape{i}{sf}");
                        }
                        else
                        {
                            sb.Append($"shape{i}");
                        }
                    }
                }
            }
            else
            {
                throw new NotSupportedException();
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

                if (langMode is LangMode.Pyb && !signMode.HasFlag(SignMode.Type))
                {
                    sb.Append("value.data(), value.length()");
                }
                else
                {
                    sb.Append("value");
                }
            }

            if (langMode is LangMode.UnCS or LangMode.UnCPP)
            {
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
                var rank = inners.Count - 1;
                sb.Append($"To{rank}DArray(value");
                for (int i = 0; i < rank; i++)
                {
                    sb.Append($", shape{i}");
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
