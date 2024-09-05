﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Reflection.Metadata;
using System.Reflection.PortableExecutable;
using System.Runtime.CompilerServices;
using System.Runtime.Loader;
using Microsoft.Extensions.Logging;

namespace Nncase.Hosting;

/// <summary>
/// Plugin loader helper.
/// </summary>
public sealed class PluginLoader
{
    public const string PluginPathEnvName = "NNCASE_PLUGIN_PATH";

    public const string ModulesDllPattern = "Nncase.Modules.*.dll";

    private static readonly string[] _builtinModules = new[]
    {
        "Nncase.Modules.StackVM.dll",
        "Nncase.Modules.CPU.dll",
        "Nncase.Modules.K210.dll",
    };

    private readonly ILogger<PluginLoader> _logger;
    private readonly AssemblyLoadContext _loadContext;

    /// <summary>
    /// Initializes a new instance of the <see cref="PluginLoader"/> class.
    /// </summary>
    /// <param name="logger">Logger.</param>
    public PluginLoader(ILogger<PluginLoader> logger)
    {
        _logger = logger;
        _loadContext = AssemblyLoadContext.GetLoadContext(typeof(PluginLoader).Assembly)
            ?? AssemblyLoadContext.Default;
    }

    public static Assembly LoadPluginAssembly(string assemblyFile, AssemblyLoadContext loadContext)
    {
        return loadContext.LoadFromAssemblyPath(assemblyFile);
    }

    public static IEnumerable<string> GetPluginAssemblies(string basePath)
    {
        if (Directory.Exists(basePath))
        {
            return (from filePath in Directory.GetFiles(basePath, ModulesDllPattern, SearchOption.AllDirectories)
                    where !_builtinModules.Contains(Path.GetFileName(filePath))
                     && IsLoadableAssembly(filePath)
                    select filePath).Distinct();
        }
        else
        {
            return Array.Empty<string>();
        }
    }

    public static IEnumerable<string> GetPluginsSearchDirectories(string pluginPathEnvName, ILogger? logger)
    {
        var directories = new List<string>();

        // 1. Environment variable
        var targetPathEnv = Environment.GetEnvironmentVariable(pluginPathEnvName);
        if (string.IsNullOrWhiteSpace(targetPathEnv))
        {
            if (logger is not null)
            {
                logger.LogWarning($"{pluginPathEnvName} is not set.");
            }
        }
        else
        {
            var targetPaths = from path in targetPathEnv!.Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries)
                              select Environment.ExpandEnvironmentVariables(path);
            directories.AddRange(targetPaths);
        }

        // 2. Python nncase modules
        var rootPath = Path.GetDirectoryName(typeof(PluginLoader).Assembly.Location)!;
        var modulesPath = Path.Combine(rootPath, "modules");
        directories.Add(modulesPath);

        if (logger is not null && logger.IsEnabled(LogLevel.Trace))
        {
            logger.LogInformation($"Loading plugins from {string.Join(", ", directories)}.");
        }

        return directories.Distinct();
    }

    public static bool IsLoadableAssembly(string filePath)
    {
        using var fs = File.OpenRead(filePath);
        using var peReader = new PEReader(fs);

        if (!peReader.HasMetadata)
        {
            return false;
        }

        var metaReader = peReader.GetMetadataReader();
        if (!metaReader.IsAssembly)
        {
            return false;
        }

        // Is reference assembly
        if ((from cah in metaReader.CustomAttributes
             let ca = metaReader.GetCustomAttribute(cah)
             where ca.Constructor.Kind == HandleKind.MemberReference
             let ctor = metaReader.GetMemberReference((MemberReferenceHandle)ca.Constructor)
             let attrType = metaReader.GetTypeReference((TypeReferenceHandle)ctor.Parent)
             where metaReader.GetString(attrType.Namespace) == nameof(System.Runtime.CompilerServices)
                && metaReader.GetString(attrType.Name) == nameof(ReferenceAssemblyAttribute)
             select cah).Any())
        {
            return false;
        }

        return true;
    }

    /// <summary>
    /// Load plugins.
    /// </summary>
    /// <returns>Plugins.</returns>
    public IReadOnlyList<IPlugin> LoadPlugins()
    {
        var pluginAsms = GetPluginsSearchDirectories(PluginPathEnvName, _logger).Select(GetPluginAssemblies).SelectMany(x => x)
                    .DistinctBy(Path.GetFileName).Select(x => LoadPluginAssembly(x, _loadContext)).Distinct().ToList();
        var plugins = (from asm in pluginAsms
                       from t in asm.ExportedTypes
                       where t.IsClass
                       && t.IsAssignableTo(typeof(IPlugin))
                       let ctor = t.GetConstructor(System.Type.EmptyTypes)
                       where ctor != null
                       select (IPlugin)ctor.Invoke(null)).ToList();

        return plugins;
    }
}
