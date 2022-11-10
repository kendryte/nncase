// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using Nncase.IR;

namespace Nncase.Hosting;
// Custom comparer for the Product class
class PathComparer : IEqualityComparer<string>
{
    // Products are equal if their names and product numbers are equal.
    public bool Equals(string x, string y)
    {
        Console.WriteLine("------------------");
        Console.WriteLine(x);
        Console.WriteLine(y);
        Console.WriteLine("------------------");
        return Path.GetFileName(x) == Path.GetFileName(y) || x == y;
    }

    // If Equals() returns true for a pair of objects
    // then GetHashCode() must return the same value for these objects.

    public int GetHashCode(string s)
    {
        return s.GetHashCode();
    }
}

/// <summary>
/// Application parts helper.
/// </summary>
public class ApplicationParts
{
    private const string _appPartsDllPattern = "Nncase.Modules.*.dll";
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
                GetApplicationPartsSearchDirectories().Select(LoadApplicationParts).SelectMany(x => x)
                    .DistinctBy(Path.GetFileName).Select(Assembly.LoadFrom))
            .Distinct().ToArray();
    }

    private static IEnumerable<string> LoadApplicationParts(string basePath)
    {
        return Directory.GetFiles(basePath, _appPartsDllPattern, SearchOption.AllDirectories)
            .Where(x => !Path.GetDirectoryName(x)!.EndsWith("ref"));
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

        foreach (var directory in directories)
        {
            Console.WriteLine(directory);
        }
        return directories.Distinct();
    }
}
