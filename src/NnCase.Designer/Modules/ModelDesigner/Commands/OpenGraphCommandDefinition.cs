using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NnCase.Designer.Commands;

namespace NnCase.Designer.Modules.ModelDesigner.Commands
{
    public class OpenGraphCommandDefinition : CommandDefinition
    {
        public override string Name => "File.OpenGraph";

        public override string Text => "Open Graph";

        public override string ToolTip => Text;
    }
}
