// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.TestFixture;

public interface IDumpDirPathProvider
{
    /// <summary>
    /// get the nncase `tests_ouput` path.
    /// <remarks>
    /// you can set the subPath for get the `xxx/tests_output/subPath`
    /// </remarks>
    /// </summary>
    /// <param name="subDir">sub directory.</param>
    /// <returns> full path string. </returns>
    public string GetDumpDirPath(string subDir);
}
