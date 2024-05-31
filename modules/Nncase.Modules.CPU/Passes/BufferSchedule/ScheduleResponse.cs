// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Text;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.TIR;

namespace Nncase.Passes.BufferSchedule;

internal sealed class ScheduledResponse
{
    private const string _bufferTypesContents = @"from dataclasses import dataclass
from enum import Enum
from typing import List
@dataclass
class Lifeness():
  start: int
  end: int

@dataclass
class Location():
  start: int
  size: int
  def __str__(self) -> str:
    return f'(start: {self.start}, size {self.size})'

@dataclass
class ScheduledBuffer():
  name: str
  lifeness: Lifeness
  location: Location
  shape: List[int]
  stride: List[int]
";

    private const string _drawContents = @"from bokeh.models import ColumnDataSource, HoverTool, FuncTickFormatter, SingleIntervalTicker, SaveTool, WheelZoomTool, WheelPanTool, ResetTool
from bokeh.palettes import Category20_20 as palette
from bokeh.plotting import figure, show
from {0} import buffers
import itertools
colors = itertools.cycle(palette)

source = {{
    ""name"": [],
    ""x"": [],
    ""y"": [],
    ""width"": [],
    ""height"": [],
    ""color"": [],
    ""location"": [],
    ""shape"":[],
    ""stride"":[],
}}

y_range_max = 0
for buffer in buffers:
  source[""name""].append(buffer.name)
  width = buffer.lifeness.end - buffer.lifeness.start
  x = buffer.lifeness.start + (width / 2)
  height = buffer.location.size
  y = buffer.location.start + (height / 2)
  y_range_max = max(y_range_max,y)
  source[""x""].append(x)
  source[""y""].append(y)
  source[""width""].append(width)
  source[""height""].append(height)
  source[""color""].append(next(colors))
  source[""location""].append(str(buffer.location))
  source[""shape""].append(','.join([str(s) for s in buffer.shape]))
  source[""stride""].append(','.join([str(s) for s in buffer.stride]))

source = ColumnDataSource(source)
hover = HoverTool(tooltips = [('name','@name'),('location','@location'),
                              ('shape','@shape'),('stride','@stride')])

p = figure(tools=[hover, WheelPanTool(), SaveTool(), WheelZoomTool(), ResetTool()], width=1280, height=720,
           y_range=(0, min(y_range_max * 2,{1})),
           title=""Local Buffer LifeTime (by Steps)"")
p.rect(x=""x"", y=""y"", width=""width"", height=""height"", fill_color=""color"", source=source)

p.yaxis.ticker = SingleIntervalTicker(interval=1024, num_minor_ticks=0)
p.yaxis.formatter = FuncTickFormatter(code=""""""
    return Math.floor(tick / (1024))
"""""")
p.ygrid.grid_line_color = 'navy'
p.ygrid.grid_line_dash = [6, 4]

p.xaxis.axis_label = ""Time (steps)""
p.outline_line_color = None

show(p)
";

    private const string _schedBufferContents = @"from buffer_types import Lifeness, Location, ScheduledBuffer
# Generator Information: {0}
buffers = [
{1}
]
";

    private readonly IReadOnlyDictionary<Expr, ScheduledBuffer> _bufferLifenessMap;

    public ScheduledResponse(
        IReadOnlyDictionary<Expr, ScheduledBuffer> bufferLifenessMap,
        bool success)
    {
        _bufferLifenessMap = bufferLifenessMap;
        Success = success;
    }

    public bool Success { get; }

    public void Dump(string file_name, string generatorInformation)
    {
        var path = Path.Combine(DumpScope.Current.Directory, "buffer_types.py");
        if (!File.Exists(path))
        {
            File.WriteAllText(path, _bufferTypesContents);
        }

        path = Path.Combine(DumpScope.Current.Directory, "draw.py");
        if (!File.Exists(path))
        {
            File.WriteAllText(path, string.Format(_drawContents, file_name, SRAM.SramSizePerThread));
        }

        var code = string.Format(
            _schedBufferContents,
            generatorInformation,
            string.Join(
                ",\n",
                _bufferLifenessMap.Select(kv => _bufferLifenessMap[kv.Key])));

        path = Path.Combine(DumpScope.Current.Directory, $"{file_name}.py");
        File.WriteAllText(path, code, System.Text.Encoding.UTF8);
    }
}
