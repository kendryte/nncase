using System;
using System.IO;

namespace Nncase.CodeGen
{
    internal static class CodeGenUtil
    {

        /// <summary>
        /// get temp file with extenstion
        /// </summary>
        /// <param name="ext"></param>
        /// <returns></returns>
        public static string GetTempFileName(string? ext = null)
        {
            ext ??= "tmp";
            if (!ext.StartsWith('.')) { ext = "." + ext; }
            return Path.GetTempPath() + Guid.NewGuid().ToString() + ext;
        }

    }
}