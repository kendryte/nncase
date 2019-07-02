using System;
using System.Diagnostics;
using System.IO;
using DotBuilder.Statements;

namespace DotBuilder
{
    public enum LayoutEngine
    {
        Dot,
        Neato,
        Twopi,
        Circos,
        Fdp,
        Sfdp,
        Patchwork,
        Osage
    }

    public enum OutputFormat
    {
        Svg,
        Bmp,
        Png,
        Jpg,
        Pdf
    }

    public class GraphViz
    {
        private readonly LayoutEngine _layoutEngine;
        private readonly OutputFormat _outputFormat;
        private readonly string _path;

        public GraphViz(string path, OutputFormat outputFormat, LayoutEngine layoutEngine = LayoutEngine.Dot)
        {
            _path = path ?? throw new ArgumentNullException(nameof(path));
            _outputFormat = outputFormat;
            _layoutEngine = layoutEngine;
        }

        public void RenderGraph(GraphBase graph, Stream outputStream)
        {
            var fileName = Path.Combine(_path, $"{_layoutEngine}.exe");
            if (!File.Exists(fileName))
                throw new Exception($@"Unable to run {fileName}. Please ensure that this the correct path to the GraphViz bin directory. If you do not have GraphVis installed then it can be downloaded from https://graphviz.gitlab.io/_pages/Download/Download_windows.html");

            var arguments = $"-T{_outputFormat.ToString().ToLower()}";
            var startInfo = new ProcessStartInfo
            {
                FileName = fileName,
                Arguments = arguments,
                UseShellExecute = false,
                CreateNoWindow = true,
                RedirectStandardInput = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true
            };


            using (var process = Process.Start(startInfo))
            {
                if (process == null)
                    throw new ArgumentNullException(nameof(process));

                using (var standardInput = process.StandardInput)
                {
                    standardInput.Write(graph.Render());
                }

                using (var standardOutput = process.StandardOutput)
                {
                    standardOutput.BaseStream.CopyTo(outputStream, 4096);
                }
            }
        }
    }
}