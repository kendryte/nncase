using System;
using System.IO;
using System.Collections.Generic;
using Nncase.IR;



namespace Nncase.CodeGen
{

    /// <summary>
    /// the runtime function
    /// </summary>
    /// <param name="Name"> func name</param>
    /// <param name="Handle"> pointer handle</param>
    public sealed record RTFunction(string Name, Delegate Handle)
    {

    }

    /// <summary>
    /// runtime module
    /// </summary>
    public interface IRTModule
    {
        /// <summary>
        /// get source code
        /// </summary>
        public string SourceText { get; }

        /// <summary>
        /// get source file ext
        /// </summary>
        public string SourceExt { get; }

        /// <summary>
        /// dump the source into file `DumpDirPath/name.xx`
        /// </summary>
        /// <param name="name">file name. </param>
        /// <param name="DumpDirPath"> dump dir path. </param>
        public void DumpSource(string name, string DumpDirPath)
        {
            using var file = File.Open($"{DumpDirPath}/{name}.{SourceExt}", FileMode.OpenOrCreate, FileAccess.Write);
            using var writer = new StreamWriter(file);
            writer.Write(SourceText);
        }

        /// <summary>
        /// compile the code
        /// </summary>
        public void Compile();

        /// <summary>
        /// call this runtime Module
        /// </summary>
        /// <param name="args"> input args</param>
        /// <returns></returns>
        public object? Invoke(params object?[]? args);

        /// <summary>
        /// get runtime function entry
        /// </summary>
        public RTFunction? Entry { get; }

        /// <summary>
        /// get the all runtime function
        /// </summary>
        public IReadOnlyList<RTFunction> Functions { get; }
    }

    /// <summary>
    /// static class for codegen collection
    /// </summary>
    public static class CodeGenExtension
    {
        /// <summary>
        /// build the RTModule
        /// </summary>
        /// <param name="mod"> input module </param>
        /// <param name="target"> target information </param>
        /// <returns> the runtime module instance </returns>
        /// <exception cref="NotImplementedException"></exception>
        public static IRTModule Build(this Module mod, Target target)
        {
            Builder.ITargetBuilder builder = target.Kind switch
            {
                TargetKindCSource csrc =>
                  csrc.Device switch
                  {
                      "Host" => new Builder.CSourceHostBuilder(),
                      _ => throw new NotImplementedException("CSource " + target.Kind.Device),
                  },
                _ => throw new NotImplementedException(target.Kind.Name),
            };
            return builder.Build(mod, target);
        }
    }
}
