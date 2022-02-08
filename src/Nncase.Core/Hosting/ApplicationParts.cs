// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;

namespace Nncase.Hosting;

/// <summary>
/// Application parts helper.
/// </summary>
public class ApplicationParts
{
    private const string _appPartsDllPattern = "Nncase*.dll";
    private const string _targetPathEnvName = "NNCASE_TARGET_PATH";

    /// <summary>
    /// Load application parts.
    /// </summary>
    /// <param name="configureAction">Configure action.</param>
    /// <returns>Application parts assemblies.</returns>
    public static Assembly[] LoadApplicationParts(Action<IList<Assembly>> configureAction)
    {
        var defaultAssemblies = new List<Assembly>() { Assembly.GetCallingAssembly() };
        configureAction(defaultAssemblies);

        return defaultAssemblies.Concat(
            GetApplicationPartsSearchDirectories().Select(LoadApplicationParts).SelectMany(x => x))
            .Distinct().ToArray();
    }

    private static IEnumerable<Assembly> LoadApplicationParts(string basePath)
    {
        return Directory.GetFiles(basePath, _appPartsDllPattern, SearchOption.AllDirectories)
            .Select(Assembly.LoadFrom);
    }

    private static IEnumerable<string> GetApplicationPartsSearchDirectories()
    {
        var directories = new List<string>();

        // 1. Executable base
        var exePath = Path.GetDirectoryName(Assembly.GetCallingAssembly().Location);
        if (!string.IsNullOrWhiteSpace(exePath))
        {
            directories.Add(exePath);
        }

        // 2. Environment variable
        var targetPathEnv = Environment.GetEnvironmentVariable(_targetPathEnvName);
        if (string.IsNullOrWhiteSpace(targetPathEnv))
        {
            // todo:log
            Console.WriteLine("NNCASE_TARGET_PATH is not set.");
        }
        else
        {
            var targetPaths = from path in targetPathEnv.Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries)
                              select Environment.ExpandEnvironmentVariables(path);
            directories.AddRange(targetPaths);
        }

        return directories.Distinct();
    }
}
