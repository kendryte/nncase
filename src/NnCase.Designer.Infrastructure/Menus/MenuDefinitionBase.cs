using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Input;
using NnCase.Designer.Commands;

namespace NnCase.Designer.Menus
{
    public abstract class MenuDefinitionBase
    {
        public abstract int SortOrder { get; }
        public abstract string Text { get; }
        public abstract Uri IconSource { get; }
        public abstract KeyGesture KeyGesture { get; }
        public abstract CommandDefinitionBase CommandDefinition { get; }
    }
}
