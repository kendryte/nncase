using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

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
        /// <param name="target">Target.</param>
        public RunPassOptions(ITarget target, int dumpLevel, string dumpDir)
        {
            Target = target;
            DumpLevel = dumpLevel;
            DumpDir = dumpDir;
            PassName = "";
        }

        /// <summary>
        /// Gets target.
        /// </summary>
        public ITarget Target { get; }

        public int DumpLevel { private set; get; }

        public string DumpDir { private set; get; }

        public string PassName { private set; get; }

        public RunPassOptions SetName(string name) { PassName = name; return this; }

        public string FullDumpDir { get => Path.Combine(DumpDir, PassName); }
    }
}
