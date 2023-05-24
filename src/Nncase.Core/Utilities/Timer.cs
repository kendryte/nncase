namespace Nncase.Utilities
{
    public class TimerRecord
    {
        private long startTime = -1;
        private double secondsElapsed = -1;
        private string _name;

        public TimerRecord(string name, TimerRecord parent)
        {
            _name = name;
        }

        public void Start()
        {
            startTime = DateTime.Now.Ticks;
        }

        public void End()
        {
            var endTime = DateTime.Now.Ticks;
            secondsElapsed = new TimeSpan(endTime - startTime).TotalSeconds;
            Console.WriteLine($"{_name} took: {secondsElapsed}");
        }
    }

    public class Timer : IDisposable
    {
        public static List<TimerRecord> Records = new();

        private TimerRecord _record;

        private Timer _parent;

        public Timer(string name, Timer? parent = null)
        {
            _record = new TimerRecord(name, null);
            Records.Add(_record);
            _record.Start();
        }

        public void Dispose()
        {
            _record.End();
        }
    }
}
