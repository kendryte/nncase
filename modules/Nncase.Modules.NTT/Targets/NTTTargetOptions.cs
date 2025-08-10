// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.Targets;

public class NTTTargetOptions : INTTTargetOptions
{
    [DisplayName("--model-name")]
    [Description("the input model name.")]
    [DefaultValue("")]
    public string ModelName { get; set; } = string.Empty;

    [DisplayName("--vectorize")]
    [Description("enable simd layout optimization.")]
    [DefaultValue(false)]
    public bool Vectorize { get; set; }

    [DisplayName("--unified-memory-arch")]
    [Description("whether Unified Memory Architecture. see https://en.wikipedia.org/wiki/Unified_Memory_Access.")]
    [DefaultValue(true)]
    public bool UnifiedMemoryArch { get; set; } = true;

    [DisplayName("--memory-access-arch")]
    [Description("Memory Access Architecture.")]
    [DefaultValue(MemoryAccessArchitecture.UMA)]
    [CommandLine.FromAmong(MemoryAccessArchitecture.UMA, MemoryAccessArchitecture.NUMA)]
    public MemoryAccessArchitecture MemoryAccessArch { get; set; } = MemoryAccessArchitecture.UMA;

    [DisplayName("--noc-arch")]
    [Description("Noc Architecture.")]
    [DefaultValue(NocArchitecture.Mesh)]
    [CommandLine.FromAmong(NocArchitecture.Mesh, NocArchitecture.CrossBar)]
    public NocArchitecture NocArch { get; set; } = NocArchitecture.Mesh;

    [DisplayName("--hierarchy-kind")]
    [Description("Hierarchy Kind.")]
    [DefaultValue(HierarchyKind.Parallel)]
    [CommandLine.FromAmong(HierarchyKind.Parallel, HierarchyKind.SMT)]
    public HierarchyKind HierarchyKind { get; set; } = HierarchyKind.Parallel;

    [DisplayName("--hierarchies")]
    [Description("the distributed hierarchies of hardware. eg. `8,4 4,8` for dynamic cluster search or `4` for fixed hardware.")]
    [DefaultValue("() => new int[][] { new int[] { 1 } }")]
    [AmbientValue("ParseNestedIntArray")]
    [CommandLine.AllowMultiplePerToken]
    public int[][] Hierarchies { get; set; } = new int[][] { new int[] { 1 } };

    [DisplayName("--hierarchy-names")]
    [Description("the name identify of hierarchies.")]
    [DefaultValue("t")]
    public string HierarchyNames { get; set; } = "t";

    [DisplayName("--hierarchy-sizes")]
    [Description("the memory capacity of hierarchies.")]
    [DefaultValue("() => new long[] { 1099511627776 }")]
    [CommandLine.AllowMultiplePerToken]
    public long[] HierarchySizes { get; set; } = new[] { 1 * (long)MathF.Pow(2, 40) };

    [DisplayName("--hierarchy-latencies")]
    [Description("the latency of hierarchies.")]
    [DefaultValue("() => new int[] { 10000 }")]
    [CommandLine.AllowMultiplePerToken]
    public int[] HierarchyLatencies { get; set; } = new[] { 10000 };

    [DisplayName("--hierarchy-bandwiths")]
    [Description("the bandwidth of hierarchies.")]
    [DefaultValue("() => new int[] { 1 }")]
    [CommandLine.AllowMultiplePerToken]
    public int[] HierarchyBandWidths { get; set; } = new[] { 1 };

    [DisplayName("--memory-capacities")]
    [Description("the memory capacity of single core. eg. `32 64` for sram,main")]
    [DefaultValue("() => new int[] { 524288, 2147483647 }")]
    [CommandLine.AllowMultiplePerToken]
    public int[] MemoryCapacities { get; set; } = new[] { 524288, int.MaxValue };

    [DisplayName("--memory-bandwidths")]
    [Description("the memory bandwidth of single core. eg. `64 8` for sram,main")]
    [DefaultValue("() => new int[] { 64, 8 }")]
    [CommandLine.AllowMultiplePerToken]
    public int[] MemoryBandWidths { get; set; } = new[] { 64, 8 };

    [DisplayName("--distributed--scheme")]
    [Description("the distributed scheme path.")]
    [DefaultValue("")]
    public string DistributedScheme { get; set; } = string.Empty;

    [DisplayName("--custom-op-scheme")]
    [Description("the custom-op scheme path.")]
    [DefaultValue("")]
    public string CustomOpScheme { get; set; } = string.Empty;
}
