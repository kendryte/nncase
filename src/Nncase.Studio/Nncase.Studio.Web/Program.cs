using Microsoft.AspNetCore.Components;
using Microsoft.AspNetCore.Components.Web;
using Nncase.Studio.Services;
using Nncase.Studio.Web.Services;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
builder.Services.AddRazorPages();
builder.Services.AddServerSideBlazor();
builder.Services.AddNncaseStudio();

builder.Services.AddScoped<IFolderPicker, FolderPicker>();

var app = builder.Build();

// Configure the HTTP request pipeline.
if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Error");
}

app.UseStaticFiles();

app.UseRouting();

app.MapBlazorHub();
app.MapFallbackToPage("/_Host");

app.Run();
