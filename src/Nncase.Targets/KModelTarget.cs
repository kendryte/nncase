using Nncase.CodeGen;
using Nncase.IR;
using Nncase.Schedule;

namespace Nncase.Targets;

/// <summary>
/// the base kmodel target for generated kmodel target file.
/// </summary>
public abstract class KModelTarget : ITarget
{
    public string Kind { get; set; } = "KModel";
    public Dictionary<string, object> Options { get; set; } = new();
    public Dictionary<string, object> Attrs { get; set; } = new();

    /// <inheritdoc/>
    public abstract void ConfigAttrs();

    /// <inheritdoc/>
    public abstract void ConfigOptions();

    /// <inheritdoc/>
    public IRTModel CreateRTModel(SchedModelResult result)
    {
        return new CodeGen.RTKModel(result, this);
    }

    /// <inheritdoc/>
    public abstract IRTModule CreateRTModule(ModuleType moduleType, SchedModuleResult ModuleResult, SchedModelResult modelResult);

    /// <inheritdoc/>
    public virtual IScheduler CreateScheduler(IRModule main_module)
    {
        return new Schedule.KModelScheduler(this, main_module);
    }
}
