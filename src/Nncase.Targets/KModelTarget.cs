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
    public IRTModel CreateRTModel(IRModel model)
    {
        return new CodeGen.RTKModel(model, this);
    }

    /// <inheritdoc/>
    public abstract IRTModule CreateRTModule(IRModel model, IRModule module);

    /// <inheritdoc/>
    public virtual IScheduler CreateScheduler(IRModule module)
    {
        return new Schedule.KModelScheduler(this, module);
    }
}
