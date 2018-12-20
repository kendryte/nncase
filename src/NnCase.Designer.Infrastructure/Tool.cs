using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NnCase.Designer.Toolbox;

namespace NnCase.Designer
{
    public abstract class Tool : LayoutItemBase, ITool
    {
        public abstract PaneLocation PreferredLocation { get; }

        public virtual double PreferredWidth => 200;

        public virtual double PreferredHeight => 200;
    }
}
