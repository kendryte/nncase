using Nncase.IR;

namespace Nncase.Schedule;

public class CSourceScheduler : IScheduler
{
    public ITarget Target { get; set; }
    public IRModule ParentModule { get; set; }

    public CSourceScheduler(IR.IRModule main_module, ITarget target)
    {
        ParentModule = main_module;
        Target = target;
    }

    public SchedModelResult Schedule(bool skip_buffer_alias = false)
    {
        return new SchedModelResult()
        {
            ParentModule = this.ParentModule,
            Modules = new(),
            Entry = new(),
        };
    }
}