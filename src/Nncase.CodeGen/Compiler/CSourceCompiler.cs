using System;
using System.IO;
using System.Text;
using System.Runtime.InteropServices;
using System.Diagnostics;


namespace Nncase.CodeGen.Compiler
{
    public class CSourceCompiler
    {
        /// <summary>
        /// compiler exe name
        /// </summary>
        string _exe = "", _arch = "", _ext = "";

        /// <summary>
        /// select current pattern's exe
        /// </summary>
        /// <exception cref="NotSupportedException"></exception>
        void PlatformSpecific()
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
            {
                _exe = "gcc";
                _ext = "so";
            }
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
            {
                _exe = "clang";
                _ext = "dylib";
            }
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                throw new NotSupportedException($"{OSPlatform.Windows}");
            }
        }

        void ArchSpecific()
        {
            _arch = RuntimeInformation.OSArchitecture switch
            {
                Architecture.X64 => "x86_64",
                Architecture.Arm64 => "arm64",
                _ => throw new NotSupportedException(RuntimeInformation.OSArchitecture.ToString()),
            };
        }

        protected string Exe
        {
            get => _exe;
        }

        protected string Arch
        {
            get => _arch;
        }

        protected string Ext
        {
            get => _ext;
        }

        public CSourceCompiler()
        {
            PlatformSpecific();
            ArchSpecific();
        }

        /// <summary>
        /// compile the source txt, write to the out_path
        /// </summary>
        /// <param name="sourcePath"> c source code</param>
        /// <param name="outPath"> out .so path </param>
        /// <returns> outPath </returns>
        public string Compile(string sourcePath, string outPath)
        {
            using (var proc = new Process())
            {
                proc.StartInfo.FileName = Exe;
                proc.StartInfo.Arguments = $"{sourcePath} -fPIC -shared -arch {Arch} -o {outPath}";
                proc.Start();
                proc.WaitForExit();
            }
            return outPath;
        }

        /// <summary>
        /// create the temp dll file and compile source
        /// <see cref="Compile(string, string)"/>
        /// </summary>
        public string Compile(string sourcePath) => Compile(sourcePath, CodeGenUtil.GetTempFileName(Ext));
    }

}