
namespace Nncase.Targets;

public class CSourceTarget : ITarget
{
    /// <inheritdoc/>
    public string Kind { get => "CSource"; set { } }
    /// <inheritdoc/>
    public Dictionary<string, object> Options { get; set; } = new();
    /// <inheritdoc/>
    public Dictionary<string, object> Attrs { get; set; } = new();
    /// <inheritdoc/>
    public void ConfigOptions() { }
    /// <inheritdoc/>
    public void ConfigAttrs() { }

    /// <inheritdoc/>
    public Schedule.IScheduler CreateScheduler(IR.IRModule main_module)
    {
        return new Schedule.CSourceScheduler(main_module, this);
    }

    /// <inheritdoc/>
    public CodeGen.IRTModel CreateRTModel(Schedule.SchedModelResult result)
    {
        return new CodeGen.CSourceRTModel(result, this);
    }

    /// <inheritdoc/>
    public CodeGen.IRTModule CreateRTModule(
      CodeGen.ModuleType moduleType,
       Schedule.SchedModuleResult ModuleResult,
        Schedule.SchedModelResult modelResult)
    {
        throw new NotImplementedException("The CSource Target Only Have Runtime Model!");
    }
}
