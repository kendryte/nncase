using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace NnCase.Runtime
{
    [StructLayout(LayoutKind.Sequential)]
    public unsafe struct ModelHeader
    {
        public const string IdentifierString = "KMDL";
        public const int IdentifierValue = 1263354956;
        public const int CurrentVersion = 4;

        public int Identifier;

        public int Version;

        public int Flags;

        public TargetId Target;

        public int ConstantUsage;

        public int MainMemoryUsage;

        public int Nodes;

        public int Inputs;

        public int Outputs;

        public int Reserved0;
    }
}
