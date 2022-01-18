// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using Nncase.IR;
using Nncase.Schedule;
using TorchSharp;
using ParameterInfo = Nncase.IR.ParameterInfo;

namespace Nncase
{
    public interface IEvaluator
    {
        public Const Visit(EvaluatorContext ctx, Op target);
    }
    
    public interface IEvaluator<T> : IEvaluator
        where T: Op
    {
        public Const Visit(EvaluatorContext ctx, T target);

        Const IEvaluator.Visit(EvaluatorContext ctx, Op target)
        {
            return Visit(ctx, (T)target);
        }
    }

    public class TargetFactory
    {
        public TargetFactory(ITarget[] targets)
        {
        }

        public ITarget GetTarget(string target)
        {
            return _targets.First(t => t.Kind == target);
        }

        private ITarget[] _targets;
    }

    public enum TargetKind
    {
        CSource,
        K210,
        K510,
        K230
    }
    
    /// <summary>
    /// Target.
    /// </summary>
    public interface ITarget
    {

        /// <summary>
        /// get the target kind.
        /// <remarks>
        /// we use the kind find the 
        /// </remarks>
        /// </summary>
        public string Kind { get; set; }

        /// <summary>
        /// Collection of Options
        /// </summary>
        public Dictionary<string, object> Options { get; set; }

        /// <summary>
        /// Collection of attributes
        /// </summary>
        public Dictionary<string, object> Attrs { get; set; }

        /// <summary>
        /// config the options
        /// </summary>
        public abstract void ConfigOptions();

        /// <summary>
        /// config the attrs
        /// </summary>
        public abstract void ConfigAttrs();

        /// <summary>
        /// get the current target schedule
        /// </summary>
        /// <param name="main_module"></param>
        /// <returns></returns>
        public abstract IScheduler CreateScheduler(IR.IRModule main_module);

        /// <summary>
        /// create the target runtime model. 
        /// <example>
        /// we will have different runtime model like CSource/KModel or other format.
        /// </example>
        /// </summary>
        /// <returns> the <see cref="CodeGen.IRTModel"/> </returns>
        public CodeGen.IRTModel CreateRTModel(Schedule.SchedModelResult result);

        /// <summary>
        /// create the target runtime module, we will call this method when build rtmodel.
        /// if you have the cross-architecture model, you can overvide this module, 
        /// eg. the k510/stackvm/k210 subclass from kmodelTarget,
        /// </summary>
        /// <returns> the module builder. </returns>
        public abstract CodeGen.IRTModule CreateRTModule(
          CodeGen.ModuleType moduleType,
           Schedule.SchedModuleResult ModuleResult,
            Schedule.SchedModelResult modelResult);

    }

    /// <summary>
    /// the Load the Plugin targets. 
    /// </summary>
    public static class PluginLoader
    {

        /// <summary>
        /// get current file path
        /// </summary>
        /// <param name="path"></param>
        /// <returns></returns>
        static string getPath([CallerFilePath] string path = null)
        {
            return path;
        }

        /// <summary>
        /// find the ITarget Class from the Assembly Dll.
        /// </summary>
        /// <param name="target_type"></param>
        /// <param name="dll_path"></param>
        /// <returns></returns>
        /// <exception cref="InvalidProgramException"></exception>
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

        /// <summary>
        /// find the dll from current project targets.
        /// </summary>
        /// <param name="target_type"></param>
        /// <param name="dll_name"></param>
        /// <returns></returns>
        static ITarget? FindFromProject(string target_type, string dll_name)
        {
            var cur_path = getPath();
            foreach (var buildType in new[] { "Debug", "Release" })
            {
                var dll_path = Path.GetFullPath(
                  Path.Combine(cur_path, "../../../targets", $"Nncase.Targets.{target_type}",
                              $"bin/{buildType}/net6.0", dll_name));
                if (File.Exists(dll_path))
                {
                    return GetTargetFromAssembly(target_type, dll_path);
                }
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
            // Setp 1. find the targets/Nncase.Target.xxx Bin directory
            var target = FindFromProject(target_type, dll_name);
            if (target is not null) return target;
            throw new InvalidProgramException($"Can't Find The Target {target_type}!");
        }
    }
}
