using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NnCase.Designer.Commands;

namespace NnCase.Designer.Modules.ModelDesigner.Commands
{
    public class ExportScriptCommandDefinition : CommandDefinition
    {
        public override string Name => "File.ExportScript";

        public override string Text => "Export Script";

        public override string ToolTip => Text;
    }
}
