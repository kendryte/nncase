// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.CodeGen.StackVM;

internal static class CodegenUtility
{
    internal static bool NormalReduceCount(TextSnippet snippet, TextSnippet input)
    {
        if (snippet.BasicBlock == input.BasicBlock)
        {
            return true;
        }

        var prevBasicBlock = snippet.BasicBlock.Prev;
        if (prevBasicBlock.Count == 1)
        {
            // if has two next, then and else has only one prev.
            var snippetInIf = prevBasicBlock[0].Nexts.Count > 1;

            // snippetInIf and snippet and input are in different BasicBlock.
            // it means input is out of if.
            if (snippetInIf)
            {
                return false;
            }
        }

        return true;
    }
}
