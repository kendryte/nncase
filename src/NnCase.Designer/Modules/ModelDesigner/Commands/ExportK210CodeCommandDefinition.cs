using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NnCase.Designer.Commands;

namespace NnCase.Designer.Modules.ModelDesigner.Commands
{
    public class ExportK210CodeCommandDefinition : CommandDefinition
    {
        public override string Name => "File.ExportK210Code";

        public override string Text => "Export K210 Code";

        public override string ToolTip => Text;
    }
}
