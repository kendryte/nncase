// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Net.NetworkInformation;
using System.Text;
using HttpMultipartParser;

namespace Nncase.Tests.CostModelTest;

internal sealed class SimulatorServer : IDisposable
{
    public static readonly HashSet<int> UsedPorts = new();

    private readonly HttpListener _listener;

    public SimulatorServer(string url)
    {
        _listener = new HttpListener();
        Url = url;
        Start();
    }

    public string Url { get; }

    public static bool GetUrl(out string url)
    {
        bool ret = false;
        url = string.Empty;
        lock (UsedPorts)
        {
            var ipProperties = IPGlobalProperties.GetIPGlobalProperties();
            IPEndPoint[] endPoints = ipProperties.GetActiveTcpListeners();
            UsedPorts.UnionWith(new HashSet<int>(endPoints.Select(p => p.Port)));
            url = string.Empty;
            for (int i = 49152; i < 65535; i++)
            {
                if (!UsedPorts.Contains(i))
                {
                    url = $"127.0.0.1:{i}";
                    UsedPorts.Add(i);
                    ret = true;
                    break;
                }
            }
        }

        return ret;
    }

    public void Dispose()
    {
        Stop();
        _listener.Close();
    }

    private void Start()
    {
        _listener.Prefixes.Add("http://" + Url + "/");
        _listener.Start();
        Receive();
    }

    private void Stop()
    {
        _listener.Stop();
    }

    private void Receive()
    {
        _listener.BeginGetContext(new AsyncCallback(ListenerCallback), _listener);
    }

    private void ListenerCallback(IAsyncResult result)
    {
        if (_listener.IsListening)
        {
            var context = _listener.EndGetContext(result);
            var request = context.Request;
            var response = context.Response;

            if (request.Url is { AbsolutePath: "/run_kmodel" } && request.HttpMethod == HttpMethod.Post.Method)
            {
                DealRunKModel(request, response);
            }
            else
            {
                WriteContent(response, "Hello World!");
                response.Close();
            }

            Receive();
        }
    }

    private void WriteContent(HttpListenerResponse response, string data)
    {
        response.ContentType = "text/html";
        response.ContentEncoding = Encoding.UTF8;
        response.ContentLength64 = Encoding.UTF8.GetByteCount(data);
        using (var strWriter = new StreamWriter(response.OutputStream, Encoding.UTF8))
        {
            strWriter.Write(data);
        }
    }

    private void DealRunKModel(HttpListenerRequest request, HttpListenerResponse response)
    {
        var contentType = request.ContentType;
        if (contentType is null)
        {
            WriteContent(response, "-1");
            response.Close();
            return;
        }

        var parser = MultipartFormDataParser.Parse(request.InputStream);

        var tempDir = Path.GetTempPath();
        var tempPaths = new List<string>();
        Directory.CreateDirectory(tempDir);
        foreach (var file in parser.Files)
        {
            var tempPath = Path.Join(tempDir, file.FileName);
            tempPaths.Add(tempPath);
            using (var of = File.OpenWrite(tempPath))
            {
                file.Data.CopyTo(of);
            }
        }

        var coutMsgBuilder = new StringBuilder();
        var errMsgBuilder = new StringBuilder();
        int exitCode = -1;
        using (var errWriter = new StringWriter(errMsgBuilder))
        {
            using (var coutWriter = new StringWriter(coutMsgBuilder))
            {
                using (var proc = new Process())
                {
                    proc.StartInfo.FileName = "nncasetest_cli";
                    proc.StartInfo.Arguments = string.Join(" ", tempPaths);
                    proc.StartInfo.RedirectStandardOutput = true;
                    proc.OutputDataReceived += (sender, e) => coutWriter.WriteLine(e.Data);
                    proc.StartInfo.RedirectStandardError = true;
                    proc.ErrorDataReceived += (sender, e) => errWriter.WriteLine(e.Data);
                    proc.Start();
                    proc.BeginOutputReadLine();
                    proc.BeginErrorReadLine();
                    proc.WaitForExit();
                    exitCode = proc.ExitCode;
                }
            }
        }

        var re = new System.Text.RegularExpressions.Regex(@"^interp run: (.*) ms");
        var countMsg = coutMsgBuilder.ToString();
        if (exitCode != 0 || re.Match(countMsg) is not System.Text.RegularExpressions.Match match)
        {
            Console.Write(errMsgBuilder);
            WriteContent(response, "-1");
            response.Close();
        }
        else
        {
            WriteContent(response, match.Groups[1].Value.ToString());
            response.Close();
        }

        foreach (var tempPath in tempPaths)
        {
            File.Delete(tempPath);
        }
    }
}
