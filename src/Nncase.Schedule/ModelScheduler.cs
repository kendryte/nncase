using Nncase.IR;

namespace Nncase.Schedule;

/// <inheritdoc/>
public class KModelScheduler : IScheduler
{
    /// <inheritdoc/>
    public ITarget Target { get; set; }
    /// <inheritdoc/>
    public IRModule Module { get; set; }

    /// <summary>
    /// create instance
    /// </summary>
    /// <param name="target"></param>
    /// <param name="module"></param>
    public KModelScheduler(ITarget target, IRModule module)
    {
        Target = target;
        Module = module;
    }

    /// <inheritdoc/>
    public IRModel Schedule(bool skip_buffer_alias = false)
    {
        throw new NotImplementedException();
    }
}