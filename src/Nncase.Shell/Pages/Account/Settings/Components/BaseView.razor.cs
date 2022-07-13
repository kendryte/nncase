using System.Threading.Tasks;
using Nncase.Shell.Models;
using Nncase.Shell.Services;
using Microsoft.AspNetCore.Components;

namespace Nncase.Shell.Pages.Account.Settings
{
    public partial class BaseView
    {
        private CurrentUser _currentUser = new CurrentUser();

        [Inject] protected IUserService UserService { get; set; }

        private void HandleFinish()
        {
        }

        protected override async Task OnInitializedAsync()
        {
            await base.OnInitializedAsync();
            _currentUser = await UserService.GetCurrentUserAsync();
        }
    }
}