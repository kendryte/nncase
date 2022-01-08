using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using Nncase.IR;
using Nncase.TIR;

namespace Nncase.CodeGen.KModel
{

    public class KModuleSerializeResult : ISerializeResult
    {
        public uint Alignment;
        public KModuleSerializeResult(uint alignment)
        {
            Alignment = alignment;
        }
    }

    public class KModelSerializeResult : ISerializeResult
    {
        public int ModelSize;
        public KModelSerializeResult(int size)
        {
            ModelSize = size;
        }
    }

    public class RTKModel : IRTModel
    {

        List<IRTModule> modules;
        string sourcePath;
        bool isSerialized;
        KModelSerializeResult serializeResult;

        public RTKModel(Schedule.SchedModelResult result, ITarget target)
        {
            Target = target;
            modelResult = result;
            Source = string.Empty;
            modules = new();
            sourcePath = CodeGenUtil.GetTempFileName(SourceExt);
            isSerialized = false;
            serializeResult = new(0);
        }

        public ITarget Target { get; set; }
        public Schedule.SchedModelResult modelResult { get; set; }
        public string Source { get; set; }
        public string SourceExt { get => "kmodel"; set { } }

        public IRTFunction? Entry { get; set; }

        public IReadOnlyList<IRTModule> Modules => modules;

        public void Dump(string name, string dumpDirPath)
        {
            if (isSerialized)
                File.Copy(sourcePath, Path.Combine(dumpDirPath, $"{name}.{SourceExt}"));
        }

        public object? Invoke(params object?[]? args)
        {
            throw new NotImplementedException();
        }

        /// <inheritdoc/>
        public ISerializeResult Serialize()
        {
            if (isSerialized) return serializeResult;

            // step 1. create the kmodel file
            using var ostream = File.Open(sourcePath, FileMode.OpenOrCreate, FileAccess.ReadWrite);
            using var writer = new BinaryWriter(ostream);
            var begin_pos = writer.BaseStream.Position;

            // step 2. start write.
            var header = new ModelHeader()
            {
                Identifier = ModelInfo.IDENTIFIER,
                Version = ModelInfo.VERSION,
                HeaderSize = (uint)Marshal.SizeOf(typeof(ModelHeader)),
                Flags = 0,
                Alignment = 8,
                Modules = (uint)modelResult.Modules.Count
            };

            var header_pos = writer.BaseStream.Position;
            writer.Seek(Marshal.SizeOf(typeof(ModelHeader)), SeekOrigin.Current);
            foreach (var moduleRes in modelResult.Modules)
            {
                var rtmodule = Target.CreateModule(moduleRes.ModuleType, moduleRes, modelResult);
                var res = (KModuleSerializeResult)rtmodule.Serialize();
                modules.Add(rtmodule);
                writer.Write(rtmodule.Source);
                header.Alignment = Math.Max(header.Alignment, res.Alignment);
            }
            // Entry point
            for (int i = 0; i < modelResult.Modules.Count; i++)
            {
                var mod_sched = modelResult.Modules[i];
                for (int j = 0; j < mod_sched.Functions.Count; j++)
                {
                    if (object.ReferenceEquals(modelResult.Entry, mod_sched.Functions[j]))
                    {
                        header.EntryModule = (uint)i;
                        header.EntryFunction = (uint)j;
                    }
                }
            }

            var end_pos = writer.BaseStream.Position;
            // write header
            writer.Seek((int)header_pos, SeekOrigin.Begin);
            writer.Write(CodeGenUtil.StructToBytes(header));
            writer.Seek((int)end_pos, SeekOrigin.Begin);
            writer.Flush();
            writer.Close();
            serializeResult.ModelSize = ((int)(end_pos - begin_pos));
            return serializeResult;
        }


    }

    // public class StackVMRTModule : IRTModule
    // {
    //     public ModuleType ModuleType { get; set; }
    //     public string Source { get; set; }
    //     public string SourceExt { get; set; }

    //     public IReadOnlyList<IRTFunction> Functions;

    //     public void Dump(string name, string dumpDirPath)
    //     {
    //         throw new NotImplementedException();
    //     }

    //     public ISerializeResult Serialize()
    //     {
    //         throw new NotImplementedException();
    //     }
    // }
}