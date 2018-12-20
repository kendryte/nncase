using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NnCase.Designer.Services;
using NnCase.Designer.Toolbox;

namespace NnCase.Designer.Modules.Toolbox
{
    public class ToolboxService : IToolboxService
    {
        private readonly Dictionary<Type, ToolboxItem[]> _toolboxItems;

        public ToolboxService(IEnumerable<ToolboxItem> toolboxItems)
        {
            _toolboxItems = (from t in toolboxItems
                             group t by t.DocumentType).ToDictionary(o => o.Key, o => o.ToArray());
        }

        public IReadOnlyList<ToolboxItem> GetToolboxItems(Type documentType)
        {
            if (_toolboxItems.TryGetValue(documentType, out var toolboxItems))
                return toolboxItems;
            return Array.Empty<ToolboxItem>();
        }
    }
}
