using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NnCase.Designer.Toolbox;

namespace NnCase.Designer.Services
{
    public interface IToolboxService
    {
        IReadOnlyList<ToolboxItem> GetToolboxItems(Type documentType);
    }
}
