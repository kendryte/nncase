using System.IO;
using System.Runtime.CompilerServices;

namespace Nncase.TestFixture;

internal sealed class DumpDirPathProvider : IDumpDirPathProvider
{
    /// <inheritdoc/>
    public string GetDumpDirPath(string subDir)
    {
        var path = Path.GetFullPath(Path.Combine(Testing.GetCallerFilePath(), "..", "..", "..", "tests_output", subDir));
        if (!Directory.Exists(path))
            Directory.CreateDirectory(path);
        return path;
    }
}