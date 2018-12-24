using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;

namespace NnCase.Designer.Modules.Inspector.Controls
{
    public class InspectorGrid
    {
        public static event EventHandler PropertyNameColumnWidthChanged;
        public static event EventHandler PropertyValueColumnWidthChanged;

        private static GridLength _propertyNameColumnWidth = new GridLength(1, GridUnitType.Star);
        public static GridLength PropertyNameColumnWidth
        {
            get { return _propertyNameColumnWidth; }
            set
            {
                _propertyNameColumnWidth = value;
                var handler = PropertyNameColumnWidthChanged;
                if (handler != null)
                    handler(null, EventArgs.Empty);
            }
        }


        private static GridLength _propertyValueColumnWidth = new GridLength(1.5, GridUnitType.Star);
        public static GridLength PropertyValueColumnWidth
        {
            get { return _propertyValueColumnWidth; }
            set
            {
                _propertyValueColumnWidth = value;
                var handler = PropertyValueColumnWidthChanged;
                if (handler != null)
                    handler(null, EventArgs.Empty);
            }
        }
    }
}
