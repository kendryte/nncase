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
using System.Threading.Tasks;
using Microsoft.CSharp;
using Nncase.CommandLine;
using RazorLight;
using RazorLight.Razor;

namespace CApiGen;

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
    private static readonly Dictionary<Type, string> Aliases =
    new Dictionary<Type, string>()
    {
        { typeof(byte), "byte" },
        { typeof(sbyte), "sbyte" },
        { typeof(short), "short" },
        { typeof(ushort), "ushort" },
        { typeof(int), "int" },
        { typeof(uint), "uint" },
        { typeof(long), "long" },
        { typeof(ulong), "ulong" },
        { typeof(float), "float" },
        { typeof(double), "double" },
        { typeof(decimal), "decimal" },
        { typeof(object), "object" },
        { typeof(bool), "bool" },
        { typeof(char), "char" },
        { typeof(string), "string" },
        { typeof(void), "void" },
    };

    public string RenderDefulatValue()
    {
        return DefulatValue switch
        {
            string v when v.StartsWith("() => ") => v,
            string v when v == string.Empty => $"() => string.Empty",
            string v => $"() => \"{v}\"",
            _ => $"() => {GetAliasValueName(DefulatValue)}",
        };
    }

    public string? GetAliasValueName(object value)
    {
        return value switch
        {
            bool b => b.ToString().ToLowerInvariant(),
            Enum e => $"{e.GetType().Name}.{e}",
            _ => value.ToString()
        };
    }

    public string OptionTypeName() => GetAliasTypeName(PropertyType);

    public string GetAliasTypeName(Type type, bool init = true)
    {
        if (Aliases.TryGetValue(type, out var alias))
        {
            return alias;
        }
        else if (type.IsEnum)
        {
            return type.Name;
        }
        else if (type.IsArray && type.GetElementType() is Type inner)
        {
            if (init)
            {
                return $"IEnumerable<{GetAliasTypeName(inner, false)}>";
            }
            else
            {
                return $"{GetAliasTypeName(inner, false)}[]";
            }
        }
        else
        {
            throw new NotSupportedException(type.FullName);
        }
    }

    public string GetUnmanagedTypeNames(bool withParam)
    {
        var sb = new StringBuilder();
        if (PropertyType.IsValueType)
        {
            sb.Append(GetAliasTypeName(PropertyType == typeof(bool) ? typeof(byte) : PropertyType));
            if (withParam)
            {
                sb.Append(' ');
                sb.Append("value");
            }
        }
        else if (PropertyType.IsArray && PropertyType.GetElementType() is Type elemType)
        {
            var elemTypeName = GetAliasTypeName(elemType);
            sb.Append($"{elemTypeName}*");
            if (withParam)
            {
                sb.Append(' ');
                sb.Append("value");
            }

            for (int i = PropertyType.GetArrayRank(); i > 0; i--)
            {
                var suffix = i == 0 ? string.Empty : "*";
                sb.Append($", nuint{suffix}");
                if (withParam)
                {
                    sb.Append(' ');
                    sb.Append($"shape{i - 1}");
                }
            }
        }
        else
        {
            throw new NotSupportedException();
        }

        return sb.ToString();
    }

    public string GetUnmanagedValueSettingNames()
    {

        var sb = new StringBuilder();
        if (PropertyType.IsValueType)
        {
            sb.Append(PropertyType == typeof(bool) ? "value != 0" : "value");
        }
        else if (PropertyType.IsArray)
        {
            sb.Append()
        }
        else
        {
            throw new NotSupportedException();
        }

        return sb.ToString();
    }
}
