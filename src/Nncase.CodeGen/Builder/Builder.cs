using System;
using Nncase.IR;

namespace Nncase.CodeGen.Builder
{
    /// <summary>
    /// Builder interface
    /// </summary>
    public interface ITargetBuilder
    {
        /// <summary>
        /// build the RTModule
        /// </summary>
        /// <param name="mod"></param>
        /// <param name="target"></param>
        /// <returns></returns>
        public RTModule Build(Module mod, Target target);
    }
}