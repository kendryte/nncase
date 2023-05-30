// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.Utilities
{
    public class TimerRecord
    {
        private readonly string _name;
        private long _startTime = -1;
        private double _secondsElapsed = -1;

        public TimerRecord(string name, TimerRecord? parent)
        {
            _name = name;
        }

        public void Start()
        {
            _startTime = DateTime.Now.Ticks;
        }

        public void End()
        {
            var endTime = DateTime.Now.Ticks;
            _secondsElapsed = new TimeSpan(endTime - _startTime).TotalSeconds;
            Console.WriteLine($"{_name} took: {_secondsElapsed}");
        }
    }

    public class Timer : IDisposable
    {
        private static readonly List<TimerRecord> _records = new();

        private readonly TimerRecord _record;

        private readonly Timer? _parent;

        public Timer(string name, Timer? parent = null)
        {
            _parent = parent;
            _record = new TimerRecord(name, null);
            _records.Add(_record);
            _record.Start();
        }

        public void Dispose()
        {
            _record.End();
        }
    }
}
