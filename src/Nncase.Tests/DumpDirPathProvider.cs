using System.IO;
using System.Runtime.CompilerServices;

namespace Nncase.TestFixture;

internal sealed class DumpDirPathProvider : IDumpDirPathProvider
{
    /// <inheritdoc/>
    public string GetDumpDirPath(string subDir)
    {
        var path = Testing.GetCallerFilePath();
        if (subDir.Length != 0)
        {
            path = Path.GetFullPath(Path.Combine(path, "..", "..", "..", "tests_output", subDir));
            if (!Directory.Exists(path))
            {
                Directory.CreateDirectory(path);
            }
        }
        return path;
    }
}