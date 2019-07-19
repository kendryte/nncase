using System;
using System.Collections.Generic;
using System.Text;
using NnCase.Runtime;
using NnCase.Runtime.Operators;

namespace NnCase.Targets.K210.Runtime.Operators
{
    public struct KPUUploadOptions
    {
        public MemoryRange Input { get; set; }

        public MemoryRange Output { get; set; }

        public RuntimeShape InputShape { get; set; }
    }

    public class KPUUploadOptionsBody : SimpleNodeBody<KPUUploadOptions>
    {
        public override RuntimeOpCode OpCode => RuntimeOpCode.K210_KPUUpload;
    }
}
