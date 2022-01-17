using Nncase.IR;

namespace Nncase.Schedule;

/// <inheritdoc/>
public class KModelScheduler : IScheduler
{
    /// <inheritdoc/>
    public ITarget Target { get; set; }
    /// <inheritdoc/>
    public IRModule ParentModule { get; set; }

    /// <summary>
    /// create instance
    /// </summary>
    /// <param name="target"></param>
    /// <param name="parent_module"></param>
    public KModelScheduler(ITarget target, IRModule parent_module)
    {
        Target = target;
        ParentModule = parent_module;
    }

    /// <inheritdoc/>
    public SchedModelResult Schedule(bool skip_buffer_alias = false)
    {
        throw new NotImplementedException();
    }
}