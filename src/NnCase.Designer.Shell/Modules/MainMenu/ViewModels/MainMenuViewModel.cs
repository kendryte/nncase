using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NnCase.Designer.Modules.MainMenu.Models;

namespace NnCase.Designer.Modules.MainMenu.ViewModels
{
    public class MainMenuViewModel : MenuModel
    {
        private readonly IMenuBuilder _menuBuilder;

        public MainMenuViewModel(IMenuBuilder menuBuilder)
        {
            _menuBuilder = menuBuilder;
            _menuBuilder.BuildMenuBar(MenuDefinictions.MainMenuBar, this);
        }
    }
}
