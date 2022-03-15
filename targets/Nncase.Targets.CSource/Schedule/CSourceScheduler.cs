using Nncase.IR;

namespace Nncase.Schedule;

public class CSourceScheduler : IScheduler
{

    public CSourceScheduler(IR.IRModule main_module, ITarget target)
    {
        Module = main_module;
        Target = target;
    }

    public ITarget Target { get; set; }
    public IRModule Module { get; set; }


    IRModel IScheduler.Schedule(bool skip_buffer_alias)
    {
        return new IRModel(new[] { Module });
    }
}