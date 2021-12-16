using System;
using System.Collections.Generic;
using Nncase.IR;

namespace Nncase.CodeGen
{

    public class RTModule
    {
        /// <summary>
        /// source code
        /// </summary>
        private string _source = "";
        /// <summary>
        /// get source code
        /// </summary>
        public virtual string Source { get => _source; set => _source = value; }
    }

    public static class CodeGen
    {
        /// <summary>
        /// build the RTModule
        /// </summary>
        /// <param name="mod"></param>
        /// <param name="target"></param>
        /// <returns></returns>
        /// <exception cref="NotImplementedException"></exception>
        public static RTModule Build(Module mod, Target target)
        {
            Builder.ITargetBuilder builder = target.Kind.Name switch
            {
                "csource" => new Builder.CSourceBuilder(),
                _ => throw new NotImplementedException(target.Kind.Name),
            };
            return builder.Build(mod, target);
        }
    }
}
