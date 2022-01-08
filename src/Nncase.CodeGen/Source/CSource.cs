using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using Nncase.IR;
using Nncase.Schedule;
using Nncase.TIR;

namespace Nncase.CodeGen.Source
{
    /// <summary>
    /// the c source runtime function.
    /// </summary>
    /// <param name="name"></param>
    /// <param name="handle"></param>
    public record CSourceRTFunction(string name, Delegate handle) : IRTFunction
    {
        public string Name { get => name; set { } }
        public Delegate Handle { get => handle; set { } }
    }

    public class CSourceSerializeResult : ISerializeResult
    {

    }

    /// <summary>
    /// c runtime module impl
    /// </summary>
    public class CSourceRTModule : IRTModule, IRTModel
    {
        /// <inheritdoc/>
        public ModuleType ModuleType { get => CodeGen.ModuleType.Create("CSource"); set { } }

        /// <inheritdoc/>
        public ITarget Target { get; set; }

        /// <inheritdoc/>
        public IReadOnlyList<IRTModule> Modules => throw new NotImplementedException();

        /// <summary>
        /// internal souce code
        /// </summary>
        string _sourcePath;

        IRModule _IRMod;
        IRTFunction? _entry = null;
        public bool IsCompiled = false;
        readonly List<IRTFunction> _functions = new();

        /// <summary>
        /// <see cref="CSourceRTModule"/>
        /// </summary>
        public CSourceRTModule(Schedule.SchedModelResult result, ITarget target)
        {
            _sourcePath = CodeGenUtil.GetTempFileName("c");
            _IRMod = result.ParentModule;
            Target = target;
        }

        /// <inheritdoc/>
        public string Source { get => File.ReadAllText(_sourcePath, Encoding.UTF8); set { } }

        /// <inheritdoc/>
        public string SourceExt { get => "c"; set { } }

        /// <inheritdoc/>
        public IRTFunction? Entry => _entry;

        /// <inheritdoc/>
        public IReadOnlyList<IRTFunction> Functions => _functions;

        public SchedModelResult modelResult { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }

        /// <inheritdoc/>
        string _dllPath = "";

        /// <inheritdoc/>
        public ISerializeResult Serialize()
        {
            if (IsCompiled) { return new CSourceSerializeResult(); }
            var compiler = new CSourceCompiler();
            _dllPath = compiler.Compile(_sourcePath);
            var dllPtr = NativeLibrary.Load(_dllPath);
            foreach (var f in _IRMod.Functions)
            {
                var funcType = f.ToDelegateType(Path.GetFileName(_dllPath));
                NativeLibrary.GetExport(dllPtr, f.Name);
                var funPtr = NativeLibrary.GetExport(dllPtr, f.Name);
                _functions.Add(new CSourceRTFunction(f.Name, funPtr.BindDelegate(funcType)));
                if (f == _IRMod.Entry) { _entry = _functions.Last(); }
            }
            return new CSourceSerializeResult();
        }

        /// <summary>
        /// invoke the module entry
        /// </summary>
        /// <param name="args">input args</param>
        /// <returns> results </returns>
        /// <exception cref="InvalidOperationException"></exception>
        public object? Invoke(params object?[]? args)
        {
            if (Entry is null)
                throw new InvalidOperationException("This RTModule Have No Entry Function!");
            return Entry.Handle.DynamicInvoke(args);
        }

        public void Dump(string name, string DumpDirPath)
        {
            using var file = File.Open($"{DumpDirPath}/{name}.{SourceExt}", FileMode.OpenOrCreate, FileAccess.Write);
            using var writer = new StreamWriter(file);
            writer.Write(Source);
        }
    }

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
            var errMsg = new StringBuilder();
            using (var errWriter = new StringWriter(errMsg))
            {
                using (var proc = new Process())
                {
                    proc.StartInfo.FileName = Exe;
                    proc.StartInfo.Arguments = $"{sourcePath} -fPIC -shared -arch {Arch} -o {outPath}";
                    proc.StartInfo.RedirectStandardError = true;
                    proc.ErrorDataReceived += (sender, e) => errWriter.WriteLine(e.Data);
                    proc.Start();
                    proc.BeginErrorReadLine();
                    proc.WaitForExit();
                    if (proc.ExitCode != 0)
                    {
                        throw new InvalidOperationException(errMsg.ToString());
                    }
                }
            }
            return outPath;
        }

        /// <summary>
        /// create the temp dll file and compile source
        /// <see cref="Compile(string, string)"/>
        /// </summary>
        public string Compile(string sourcePath) => Compile(sourcePath, CodeGenUtil.GetTempFileName(Ext));
    }


    /// <summary>
    /// the builder dispatcher
    /// </summary>
    // public class CSourceHostBuilder : IModelBuilder
    // {
    //     string _sourcePath;
    //     public CSourceHostBuilder()
    //     {
    //         _sourcePath = CodeGenUtil.GetTempFileName("c");
    //     }

    //     /// <inheritdoc/>
    //     public IRTModule Build(IRModule mod, Target target)
    //     {

    //         using (var file = File.Open(_sourcePath, FileMode.OpenOrCreate, FileAccess.Write))
    //         {
    //             using (var writer = new StreamWriter(file))
    //             {
    //                 var visior = new CSourceHostBuildVisior(writer);
    //                 if (mod.Entry is null) { throw new InvalidProgramException("The Model Entry Is Null!"); }
    //                 if (mod.Entry.CheckedType is null && mod.Entry.InferenceType() == false) { throw new InvalidProgramException("The Model Entry Can't Inference Type!"); }
    //                 visior.Visit(mod.Entry);
    //             }
    //         }
    //         var rt = new CSourceRTModule(mod, _sourcePath);
    //         return rt;
    //     }
    // }
}