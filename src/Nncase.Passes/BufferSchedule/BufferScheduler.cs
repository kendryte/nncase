// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reactive;
using Nncase;
using Nncase.IR;

namespace Nncase.Passes.BufferSchedule;

internal sealed class BufferScheduler
{
    public List<ScheduleBuffer> CollectLifeTime(Function func)
    {
        var c = new LifeTimeCollector();
        return c.Collect(func);
    }

    public void DumpScheduled(string path, List<ScheduleBuffer> buffers)
    {
        using (var fs = File.OpenWrite(path))
        {
            using (var wr = new StreamWriter(fs))
            {
                wr.Write(@"from bokeh.models import ColumnDataSource, HoverTool, FuncTickFormatter, SingleIntervalTicker, SaveTool, WheelZoomTool, WheelPanTool, ResetTool
from bokeh.palettes import Category20_20 as palette
from bokeh.plotting import figure, show, save
import itertools
from dataclasses import dataclass
from enum import Enum
from typing import List
@dataclass
class Interval():
  start: int
  end: int

@dataclass
class Location():
  depth_start: int
  depth_size: int
  def __str__(self) -> str:
    return f'(start: {self.depth_start}, size {self.depth_size})'

class ConstraintsMode(Enum):
  No = 0
  Channel = 1

@dataclass
class ScheduledBuffer():
  name: str
  interval: Interval
  location: Location
  constraints: ConstraintsMode
  shape: List[int]
  stride: List[int]

colors = itertools.cycle(palette)

buffers = [
");
                foreach (var item in buffers)
                {
                    wr.WriteLine(item.ToString());
                }

                wr.Write(@"]

source = {
    'name': [],
    'x': [],
    'y': [],
    'width': [],
    'height': [],
    'color': [],
    'location': [],
    'shape': [],
    'stride': [],
}

y_range_max = 0
for buffer in buffers:
  source['name'].append(buffer.name)
  width = buffer.interval.end - buffer.interval.start
  x = buffer.interval.start + (width // 2)
  height = buffer.location.depth_size
  y = buffer.location.depth_start + (height // 2)
  y_range_max = max(y_range_max, y)
  source['x'].append(x)
  source['y'].append(y)
  source['width'].append(width)
  source['height'].append(height)
  source['color'].append(next(colors))
  source['location'].append(str(buffer.location))
  source['shape'].append(','.join([str(s) for s in buffer.shape]))
  source['stride'].append(','.join([str(s) for s in buffer.stride]))

source = ColumnDataSource(source)
hover = HoverTool(tooltips=[('name', '@name'), ('location', '@location'),
                            ('shape', '@shape'), ('stride', '@stride')])

p = figure(tools=[hover, WheelPanTool(), SaveTool(), WheelZoomTool(), ResetTool()], width=1280, height=720,
           y_range=(0, y_range_max * 2),
           title='Local Buffer LifeTime (by Steps)')
p.rect(x='x', y='y', width='width', height='height', fill_color='color', source=source)
p.xaxis.axis_label = 'Time (steps)'
p.outline_line_color = None

show(p)");
            }
        }
    }
}
