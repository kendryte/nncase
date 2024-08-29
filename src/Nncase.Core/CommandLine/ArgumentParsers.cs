// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.CommandLine;

public static class ArgumentParsers
{
    public static IEnumerable<int[]> ParseNestedIntArray(System.CommandLine.Parsing.ArgumentResult result)
    {
        return result.Tokens.Select(tk => tk.Value.Split(",").Select(i => int.Parse(i)).ToArray());
    }
}
