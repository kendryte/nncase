using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.Transform
{
    /// <summary>
    /// Options for running pass.
    /// </summary>
    public class RunPassOptions
    {
        /// <summary>
        /// constructor
        /// </summary>
        /// <param name="target"> target device </param>
        /// <param name="dumpLevel"> int level </param>
        /// <param name="dumpDir"> dir </param>
        public RunPassOptions(ITarget target, int dumpLevel, string dumpDir)
        {
            Target = target;
            DumpLevel = dumpLevel;
            DumpDir = dumpDir;
            PassName = "";
        }
        /// <summary>
        /// copy construct
        /// </summary>
        /// <param name="other"></param>
        public RunPassOptions(RunPassOptions other)
        {
            Target = other.Target;
            DumpLevel = other.DumpLevel;
            DumpDir = other.DumpDir;
            PassName = other.PassName;
        }


        /// <summary>
        /// Gets target.
        /// </summary>
        public ITarget Target { get; }

        /// <summary>
        /// Dump level 0 = do nothing
        /// Dump level 1 = print to std output
        /// Dump level 2 = print dump to file
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
        /// set the pass name
        /// </summary>
        /// <param name="name"></param>
        /// <returns></returns>
        public RunPassOptions SetName(string name) { PassName = name; return this; }

        /// <summary>
        /// set the dumpDir
        /// </summary>
        /// <param name="path"></param>
        /// <returns></returns>
        public RunPassOptions SetDir(string path) { DumpDir = path; return this; }

        /// <summary>
        /// return "{DumpDir}/{PassName}"
        /// </summary>
        public string FullDumpDir { get => Path.Combine(DumpDir, PassName); }

        /// <summary>
        /// the invalid pass 
        /// </summary>
        public static RunPassOptions Invalid => new RunPassOptions(null!, -1, "");
    }
}
