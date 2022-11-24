// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.PatternMatch;

namespace Nncase.Transform
{
    /// <summary>
    /// Options for running pass.
    /// </summary>
    public class RunPassOptions
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="RunPassOptions"/> class.
        /// </summary>
        /// <param name="compileOptions"></param>
        public RunPassOptions(CompileOptions compileOptions)
        {
            Target = CompilerServices.GetTarget(compileOptions.Target);
            DumpLevel = compileOptions.DumpLevel;
            DumpDir = compileOptions.DumpDir;
            CompileOptions = compileOptions;
            PassName = "";
        }

        /// <summary>
        /// parameterless ctor.
        /// </summary>
        public RunPassOptions(ITarget target)
        {
            Target = target;
            DumpLevel = CompilerServices.CompileOptions.DumpLevel;
            DumpDir = CompilerServices.CompileOptions.DumpDir;
            CompileOptions = CompilerServices.CompileOptions;
            PassName = "";
        }

        /// <summary>
        /// constructor.
        /// </summary>
        /// <param name="target"> target device. </param>
        /// <param name="dumpLevel"> int level. </param>
        /// <param name="dumpDir"> dir. </param>
        public RunPassOptions(ITarget target, int dumpLevel, string dumpDir)
            : this(target, dumpLevel, dumpDir, CompilerServices.CompileOptions)
        {
        }

        /// <summary>
        /// create the run pass options
        /// </summary>
        /// <param name="target"></param>
        /// <param name="dumpLevel"></param>
        /// <param name="dumpDir"></param>
        /// <param name="options"></param>
        public RunPassOptions(ITarget target, int dumpLevel, string dumpDir, CompileOptions options)
        {
            Target = target;
            DumpLevel = dumpLevel;
            DumpDir = dumpDir;
            PassName = "";
            RewriteOnce = false;
            CompileOptions = options;
        }

        /// <summary>
        /// copy construct.
        /// </summary>
        /// <param name="other"></param>
        public RunPassOptions(RunPassOptions other)
        {
            Target = other.Target;
            DumpLevel = other.DumpLevel;
            DumpDir = other.DumpDir;
            PassName = other.PassName;
            RewriteOnce = other.RewriteOnce;
            CompileOptions = other.CompileOptions;
        }

        /// <summary>
        /// Gets target.
        /// </summary>
        public ITarget Target { get; }

        /// <summary>
        /// Dump level 0 = do nothing
        /// Dump level 1 = print to std output
        /// Dump level 2 = print dump to file.
        /// </summary>
        public int DumpLevel { private set; get; }

        /// <summary>
        /// Dump dir
        /// </summary>
        public string DumpDir { private set; get; }

        /// <summary>
        /// current pass name
        /// </summary>
        public string PassName { private set; get; }

        /// <summary>
        /// Control rewrite once or not.
        /// Default is false.
        /// </summary>
        public bool RewriteOnce { private set; get; }

        /// <summary>
        /// get the compile options
        /// </summary>
        public CompileOptions CompileOptions { set; get; }

        /// <summary>
        /// the match option.
        /// </summary>
        public MatchOptions MatchOptions { get; set; } = new MatchOptions();

        /// <summary>
        /// set the pass name
        /// </summary>
        /// <param name="name"></param>
        /// <returns></returns>
        public RunPassOptions SetPassName(string name) => new(Target, DumpLevel, DumpDir, CompileOptions) { PassName = name, MatchOptions = MatchOptions };

        /// <summary>
        /// set the dumpDir.
        /// </summary>
        /// <param name="path"></param>
        /// <returns></returns>
        public RunPassOptions SetDumpDir(string path) => new(Target, DumpLevel, path, CompileOptions) { PassName = PassName, MatchOptions = MatchOptions };

        /// <summary>
        /// set the dump level
        /// </summary>
        /// <param name="dumpLevel"></param>
        /// <returns></returns>
        public RunPassOptions SetDumpLevel(int dumpLevel) => new(Target, dumpLevel, DumpDir, CompileOptions) { PassName = PassName, MatchOptions = MatchOptions };

        /// <summary>
        /// set the RewriteOnce
        /// </summary>
        /// <param name="once"></param>
        /// <returns></returns>
        public RunPassOptions SetRewriteOnce(bool once) => new(Target, DumpLevel, DumpDir, CompileOptions) { PassName = PassName, RewriteOnce = once, MatchOptions = MatchOptions };

        /// <summary>
        /// indent the dumpDir.
        /// </summary>
        /// <param name="path"></param>
        /// <returns></returns>
        public RunPassOptions IndentDir(string path) => new(Target, DumpLevel, Path.Combine(DumpDir, path), CompileOptions) { PassName = PassName, MatchOptions = MatchOptions };

        /// <summary>
        /// return "{DumpDir}/{PassName}".
        /// </summary>
        public string PassDumpDir { get => Path.Combine(DumpDir, PassName); }

        /// <summary>
        /// the invalid pass 
        /// </summary>
        public static RunPassOptions Invalid => new RunPassOptions(null!, -1, "");
    }
}
