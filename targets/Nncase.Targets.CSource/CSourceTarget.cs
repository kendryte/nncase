
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
    public CodeGen.IRTModel CreateRTModel(IR.IRModel model)
    {
        return new CodeGen.CSourceRTModel(model, this);
    }

    /// <inheritdoc/>
    public CodeGen.IRTModule CreateRTModule(IR.IRModel model, IR.IRModule module)
    {
        throw new NotImplementedException("The CSource Target Only Have Runtime Model!");
    }
}
