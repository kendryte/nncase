using Nncase.IR;
namespace Nncase.CodeGen
{
    /// <summary>
    /// static class for codegen collection
    /// </summary>
    public static class CodeGenExtension
    {
        /// <summary>
        /// schedule and build the IRModule to RTModel
        /// </summary>
        /// <param name="mod"> input module </param>
        /// <param name="target"> target information </param>
        /// <returns> the runtime model instance </returns>
        public static IRTModel Build(this IRModule mod, ITarget target)
        {
            var sch = target.CreateScheduler(mod, target);
            var schr = sch.Schedule();
            return target.CreateModel(schr);
        }
    }
}