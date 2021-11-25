using Xunit;
using System;
using System.Collections.Generic;
using Nncase.IR;
using Nncase.Transform;
using Nncase.Pattern;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.F.NN;
using Rule = Nncase.Transform.Rule;
using static Nncase.Pattern.F.Math;
using static Nncase.Pattern.F.NN;
using static Nncase.Pattern.F.Tensors;
using static Nncase.Pattern.Utility;
using System.IO;
using System.Runtime.CompilerServices;

namespace Nncase.Tests.ReWrite
{
    public class RewriteTest
    {
        public RunPassOptions passOptions;

        private static string GetThisFilePath([CallerFilePath] string path = null)
        {
            return path;
        }

        public RewriteTest()
        {
            var TestName = this.GetType().Name;
            string dumpDir = Path.Combine(GetThisFilePath(), "..", "..", "..", "..", "tests_output");
            dumpDir = Path.GetFullPath(dumpDir);
            Directory.CreateDirectory(dumpDir);
            passOptions = new RunPassOptions(null, 3, dumpDir);
        }
    }
}