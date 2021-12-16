// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;

namespace Nncase
{
    /// <summary>
    /// The device type in Device.
    /// </summary>
    public enum DeviceType
    {
        /// <summary>
        /// CPU device 
        /// </summary>
        CPU,
        /// <summary>
        /// CUDA GPU device 
        /// </summary>
        CUDA,
        /// <summary>
        /// Pinned CUDA CPU memory by cudaMallocHost
        /// </summary>
        CUDAHost,
        /// <summary>
        /// OpenCL devices. 
        /// </summary>
        OpenCL,
        /// <summary>
        /// Vulkan buffer for next generation graphics. 
        /// </summary>
        Vulkan,
        /// <summary>
        /// Metal for Apple GPU. 
        /// </summary>
        Metal,
        /// <summary>
        /// Verilog simulator buffer 
        /// </summary>
        VPI,
        /// <summary>
        /// ROCm GPUs for AMD GPUs 
        /// </summary>
        ROCM,
        /// <summary>
        /// Reserved extension device type,
        /// used for quickly test extension device
        /// The semantics can differ depending on the implementation.
        /// </summary>
        ExtDev,
    }

    /// <summary>
    /// Target kind, specifies the kind of the target
    /// </summary>
    public abstract class TargetKindBase
    {
        /// <summary>
        ///  Name of the target kind
        /// </summary>
        public string Name;
        /// <summary>
        ///  Device type of target kind
        /// </summary>
        public DeviceType DeviceType;
        /// <summary>
        ///  Default keys of the target
        /// </summary>
        public List<string> DefaultKeys;
        /// <summary>
        ///  Function used to preprocess on target creation
        /// </summary>
        public Action<int> Preprocessor;

        public TargetKindBase(string name, DeviceType deviceType)
        {
            Name = name;
            DeviceType = deviceType;
            DefaultKeys = new();
            void fun(int x) { }
            Preprocessor = fun;
        }
    }

    public class TargetKind : TargetKindBase
    {
        /// <summary>
        /// keys
        /// </summary>
        public List<string>? DefulatKeys;
        /// <summary>
        /// tag
        /// </summary>
        public string? Tag;
        /// <summary>
        /// device
        /// </summary>
        public string? Device;
        /// <summary>
        /// dl model
        /// </summary>
        public string? Model;
        /// <summary>
        /// use libs
        /// </summary>
        public List<string>? Libs;
        /// <summary>
        /// host taget
        /// </summary>
        public Target? Host;
        /// <summary>
        /// from smoe device
        /// </summary>
        public int? FromDevice;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="name"></param>
        /// <param name="deviceType"></param>
        /// <param name="default_keys"></param>
        /// <param name="tag"></param>
        /// <param name="device"></param>
        /// <param name="model"></param>
        /// <param name="libs"></param>
        /// <param name="host"></param>
        /// <param name="from_device"></param>
        public TargetKind(string name, DeviceType deviceType, List<string>? default_keys, string? tag, string? device, string? model,
                         List<string>? libs, Target? host, int? from_device) : base(name, deviceType)
        {
            DefulatKeys = default_keys;
            Tag = tag;
            Device = device;
            Model = model;
            Libs = libs;
            Host = host;
            FromDevice = from_device;
        }

    }


    public class TargetKindLLVM : TargetKind
    {
        public List<string>? Mattr;
        public string? Mcpu;
        public string? Mtriple;
        public string? MfloatAbi;
        public string? Mabi;
        public bool? SystemLib;
        public string? Runtime;
        public bool? LinkParams;
        public bool? UnpackedApi;
        public string? InterfaceApi;
        public bool? FastMath;
        public bool? FastMathNnan;
        public bool? FastMathNinf;
        public bool? FastMathNsz;
        public bool? FastMathArcp;
        public bool? FastMathContract;
        public bool? FastMathReassoc;
        public int? OptLevel;

        /// <summary>
        /// consturct targetkind llvm
        /// </summary>
        /// <param name="keys"></param>
        /// <param name="tag"></param>
        /// <param name="device"></param>
        /// <param name="model"></param>
        /// <param name="libs"></param>
        /// <param name="host"></param>
        /// <param name="from_device"></param>
        /// <param name="mattr"></param>
        /// <param name="mcpu"></param>
        /// <param name="mtriple"></param>
        /// <param name="mfloat_abi"></param>
        /// <param name="mabi"></param>
        /// <param name="system_lib"></param>
        /// <param name="runtime"></param>
        /// <param name="link_params"></param>
        /// <param name="unpacked_api"></param>
        /// <param name="interface_api"></param>
        /// <param name="fast_math"></param>
        /// <param name="fast_math_nnan"></param>
        /// <param name="fast_math_ninf"></param>
        /// <param name="fast_math_nsz"></param>
        /// <param name="fast_math_arcp"></param>
        /// <param name="fast_math_contract"></param>
        /// <param name="fast_math_reassoc"></param>
        /// <param name="opt_level"></param>
        public TargetKindLLVM(List<string>? keys = null, string? tag = null, string? device = null, string? model = null, List<string>? libs = null, Target? host = null, int? from_device = null,
          List<string>? mattr = null, string? mcpu = null, string? mtriple = null, string? mfloat_abi = null, string? mabi = null, bool? system_lib = null, string? runtime = null, bool? link_params = false, bool? unpacked_api = null, string? interface_api = null, bool? fast_math = null, bool? fast_math_nnan = null, bool? fast_math_ninf = null, bool? fast_math_nsz = null, bool? fast_math_arcp = null, bool? fast_math_contract = null, bool? fast_math_reassoc = null, int? opt_level = null) : base("llvm", DeviceType.CPU, new() { "cpu" }, tag, device, model, libs, host, from_device)
        {
            Mattr = mattr;
            Mcpu = mcpu;
            Mtriple = mtriple;
            MfloatAbi = mfloat_abi;
            Mabi = mabi;
            SystemLib = system_lib;
            Runtime = runtime;
            LinkParams = link_params;
            UnpackedApi = unpacked_api;
            InterfaceApi = interface_api;
            FastMath = fast_math;
            FastMathNnan = fast_math_nnan;
            FastMathNinf = fast_math_ninf;
            FastMathNsz = fast_math_nsz;
            FastMathArcp = fast_math_arcp;
            FastMathContract = fast_math_contract;
            FastMathReassoc = fast_math_reassoc;
            OptLevel = opt_level;
        }
    }

    /// <summary>
    /// Target.
    /// </summary>
    public sealed class Target
    {
        /// <summary>
        /// The kind of the target device
        /// </summary>
        public TargetKind Kind { get; }

        /// <summary>
        /// Target host information, must be Target type
        /// </summary>
        Target? Host { get; }

        /// <summary>
        /// Tag of the the target, can be empty
        /// </summary>
        string Tag { get; }

        /// <summary>
        /// Keys for this target
        /// </summary>
        public List<string> Keys { get; }

        /// <summary>
        /// Collection of attributes
        /// </summary>
        Dictionary<string, object> Attrs { get; }


        /// <summary>
        /// constructor for target
        /// </summary>
        /// <param name="kind"></param>
        /// <param name="keys"></param>
        /// <param name="attrs"></param>
        /// <param name="tag"></param>
        /// <param name="host"></param>
        public Target(TargetKind kind, List<string> keys, Dictionary<string, object> attrs, string tag = "", Target? host = null)
        {
            Kind = kind;
            Keys = keys;
            Attrs = attrs;
            Host = host;
            Tag = tag;
        }

        /// <summary>
        /// Returns a ARM CPU target.
        /// This function will also download pre-tuned op parameters when there is none.
        /// </summary>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Target ArmCpu(string name)
        {
            Dictionary<string, TargetKind> table = new() { { "m1", new TargetKindLLVM(mtriple: "arm64-apple-darwin21.1.0") } };
            if (!table.TryGetValue(name, out var targetKind))
            {
                throw new InvalidOperationException($"Can't Find Cpu {name}!");
            }
            targetKind.Device = "arm_cpu";
            return new(targetKind, new() { "cpu", "arm_cpu" }, new());
        }
    }
}
