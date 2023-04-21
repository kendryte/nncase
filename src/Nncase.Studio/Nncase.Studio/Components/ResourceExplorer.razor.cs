// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AntDesign;
using Microsoft.AspNetCore.Components;
using Microsoft.Extensions.Logging;

namespace Nncase.Studio.Components;

public partial class ResourceExplorer
{
    [Inject]
    public ILogger<ResourceExplorer> Logger { get; set; } = default!;

    [Parameter]
    public string WorkingDirectoy { get; set; } = default!;

    [Parameter]
    public EventCallback<string> FileSelected { get; set; }

    private IReadOnlyList<ResourceNode>? TopNodes { get; set; }

    protected override void OnInitialized()
    {
        LoadDirectoryContent();
    }

    private Task OnNodeSelected(TreeEventArgs<ResourceNode> e)
    {
        return FileSelected.InvokeAsync(e.Node.DataItem.Path);
    }

    private void LoadDirectoryContent()
    {
        var directory = new DirectoryInfo(WorkingDirectoy);
        var nodes = directory.EnumerateFileSystemInfos().Select(BuildNode).ToArray();
        TopNodes = nodes;
    }

    private ResourceNode BuildNode(FileSystemInfo entry)
    {
        var node = new ResourceNode(entry.Name, entry.FullName, entry.Attributes.HasFlag(FileAttributes.Directory));
        if (entry is DirectoryInfo directory)
        {
            FillNodeChildren(node, directory);
        }

        return node;
    }

    private void FillNodeChildren(ResourceNode node, DirectoryInfo directory)
    {
        try
        {
            foreach (var child in directory.EnumerateFileSystemInfos())
            {
                node.Children.Add(BuildNode(child));
            }
        }
        catch (Exception ex)
        {
            Logger.LogError(ex, ex.Message);
        }
    }
}

internal sealed class ResourceNode
{
    public ResourceNode(string name, string path, bool isDirectory)
    {
        Name = name;
        Path = path;
        IsDirectory = isDirectory;
    }

    public string Name { get; set; }

    public string Path { get; set; }

    public bool IsDirectory { get; set; }

    public List<ResourceNode> Children { get; } = new();

    public string Icon => IsDirectory ? "folder" : GetIconFromFileName(Name);

    private static string GetIconFromFileName(string name)
    {
        return "file";
    }
}
