using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NnCase.Designer.Commands
{
    public abstract class CommandDefinition : CommandDefinitionBase
    {
        public override Uri IconSource => null;

        public sealed override bool IsList => false;
    }
}
