// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Net;
using System.Net.Http;
using System.Text;
using HttpMultipartParser;

namespace Nncase.Tests.CostModelTest;


internal sealed class SimulatorServer : IDisposable
{
    private HttpListener _listener;

    public SimulatorServer(string url)
    {
        _listener = new HttpListener();
        Url = url;
        Start();
    }

    public string Url { get; }

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
            var resp = context.Response;

            if (request.Url is { AbsolutePath: "/run_kmodel" } && request.HttpMethod == HttpMethod.Post.Method)
            {
                DealRunKModel(request, resp);
            }
            else
            {
                byte[] data = Encoding.UTF8.GetBytes("Hello World!");
                resp.ContentType = "text/html";
                resp.ContentEncoding = Encoding.UTF8;
                resp.ContentLength64 = data.LongLength;

                // Write out to the response stream (asynchronously), then close it
                resp.OutputStream.Write(data, 0, data.Length);
                resp.Close();
            }


            Receive();
        }
    }

    private void DealRunKModel(HttpListenerRequest request, HttpListenerResponse response)
    {
        var contentType = request.ContentType;
        if (contentType is null)
        {
            byte[] data = Encoding.UTF8.GetBytes("-1");
            response.ContentType = "text/html";
            response.ContentEncoding = Encoding.UTF8;
            response.ContentLength64 = data.LongLength;
            // Write out to the response stream (asynchronously), then close it
            response.OutputStream.Write(data, 0, data.Length);
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
            var of = File.OpenWrite(tempPath);
            file.Data.CopyTo(of);
        }

        var errMsg = new StringBuilder();
        long ms = 0;
        using (var errWriter = new StringWriter(errMsg))
        {
            var watch = System.Diagnostics.Stopwatch.StartNew();
            using (var proc = new Process())
            {
                proc.StartInfo.FileName = "nncasetest_cli";
                proc.StartInfo.Arguments = string.Join(" ", tempPaths);
                proc.StartInfo.RedirectStandardError = true;
                proc.ErrorDataReceived += (sender, e) => errWriter.WriteLine(e.Data);
                proc.Start();
                proc.BeginErrorReadLine();
                proc.WaitForExit();
                if (proc.ExitCode != 0)
                {
                    throw new InvalidOperationException(errMsg.ToString());
                }
            }
            watch.Stop();
            ms = watch.ElapsedMilliseconds;
        }

        if (errMsg.Length > 0)
        {
            byte[] data = Encoding.UTF8.GetBytes("-1");
            response.ContentType = "text/html";
            response.ContentEncoding = Encoding.UTF8;
            response.ContentLength64 = data.LongLength;
            response.OutputStream.Write(data, 0, data.Length);
            response.Close();
        }
        else
        {
            byte[] data = Encoding.UTF8.GetBytes(ms.ToString());
            response.ContentType = "text/html";
            response.ContentEncoding = Encoding.UTF8;
            response.ContentLength64 = data.LongLength;
            response.OutputStream.Write(data, 0, data.Length);
            response.Close();
        }
    }
}