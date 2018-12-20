using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NnCase.Designer.Toolbox;

namespace NnCase.Designer
{
    public interface ITool : ILayoutItem
    {
        PaneLocation PreferredLocation { get; }

        double PreferredWidth { get; }

        double PreferredHeight { get; }
    }
}
