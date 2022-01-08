// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Runtime.CompilerServices;
using Nncase.Schedule;

namespace Nncase
{
    /// <summary>
    /// Target.
    /// </summary>
    public interface ITarget
    {

        /// <summary>
        /// get the target kind.
        /// </summary>
        public string Kind { get; set; }

        /// <summary>
        /// Collection of Options
        /// </summary>
        Dictionary<string, object> Options { get; set; }

        /// <summary>
        /// Collection of attributes
        /// </summary>
        Dictionary<string, object> Attrs { get; set; }

        /// <summary>
        /// config the options
        /// </summary>
        public void ConfigOptions();

        /// <summary>
        /// config the attrs
        /// </summary>
        public void ConfigAttrs();

        /// <summary>
        /// get the current target schedule
        /// </summary>
        /// <param name="main_module"></param>
        /// <param name="target"></param>
        /// <returns></returns>
        public IScheduler CreateScheduler(IR.IRModule main_module, ITarget target);

        /// <summary>
        /// create the target runtime model. 
        /// <example>
        /// we will have different runtime model like CSource/KModel or other format.
        /// </example>
        /// </summary>
        /// <returns> the <see cref="CodeGen.IRTModel"/> </returns>
        public CodeGen.IRTModel CreateModel(Schedule.SchedModelResult result);

        /// <summary>
        /// create the target runtime module
        /// <example>
        /// we will have k510/stackvm/k210 module.
        /// </example>
        /// </summary>
        /// <returns> the module builder. </returns>
        public abstract CodeGen.IRTModule CreateModule(
          CodeGen.ModuleType moduleType,
           Schedule.SchedModuleResult ModuleResult,
            Schedule.SchedModelResult modelResult);

    }

    public static class PluginLoader
    {

        static string getPath([CallerFilePath] string path = null)
        {
            return path;
        }

        static ITarget GetTargetFromAssembly(string target_type, string dll_path)
        {
            var assembly = Assembly.LoadFrom(dll_path);
            var targetType = typeof(ITarget);
            foreach (var eType in assembly.ExportedTypes)
            {
                if (targetType.IsAssignableFrom(eType))
                {
                    var target = Activator.CreateInstance(eType);
                    if (target is not null)
                    {
                        return (ITarget)target;
                    }
                    else
                    {
                        throw new InvalidProgramException($"Can't Create The Instance From {eType.Name}");
                    }
                }
            }
            throw new InvalidProgramException($"Can't Find The Derived Target Class From {dll_path}!");
        }

        static ITarget? FindFromProject(string target_type, string dll_name)
        {
            var cur_path = getPath();
            var dll_path = Path.GetFullPath(Path.Combine(cur_path, "../..", $"Nncase.Targets.{target_type}", "bin/Debug/net6.0", dll_name));
            if (File.Exists(dll_path))
            {
                return GetTargetFromAssembly(target_type, dll_path);
            }
            return null;
        }

        /// <summary>
        /// load the target from the dll
        /// </summary>
        /// <param name="target_type"></param>
        /// <returns></returns>
        public static ITarget CreateTarget(string target_type)
        {
            var dll_name = $"Nncase.Targets.{target_type}.dll";
            // Setp 1. find the Nncase.Target Bin directory
            var target = FindFromProject(target_type, dll_name);
            if (target is not null) return target;
            throw new InvalidProgramException($"Can't Find The Target {target_type}!");
        }
    }
}
