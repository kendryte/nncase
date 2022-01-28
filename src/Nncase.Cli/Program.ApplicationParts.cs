// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.Cli
{
    internal partial class Program
    {
        private const string _appPartsDllPattern = "Nncase*.dll";
        private const string _targetPathEnvName = "NNCASE_TARGET_PATH";

        private static readonly IEnumerable<Assembly> _defaultAssemblies = new[] { typeof(Program).Assembly };

        private static IEnumerable<Assembly> LoadApplicationParts(string basePath)
        {
            return Directory.GetFiles(basePath, _appPartsDllPattern, SearchOption.AllDirectories)
                .Select(Assembly.LoadFile);
        }

        private static IEnumerable<string> GetApplicationPartsSearchDirectories()
        {
            var directories = new List<string>();

            // 1. Executable base
            var exePath = Path.GetDirectoryName(Assembly.GetEntryAssembly().Location);
            directories.Add(exePath);

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

        private static Assembly[] LoadApplicationParts()
        {
            return GetApplicationPartsSearchDirectories().Select(LoadApplicationParts)
                .SelectMany(x => x).Distinct().ToArray();
        }
    }
}
