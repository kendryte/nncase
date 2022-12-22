// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

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
        {
            Directory.CreateDirectory(path);
        }

        return path;
    }
}
