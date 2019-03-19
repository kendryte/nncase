using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace NnCase.Converter.K210.Emulator
{
    public class K210Emulator
    {
        private readonly byte[] _kmodel;

        public K210Emulator(byte[] kmodel)
        {
            _kmodel = kmodel;
        }
    }
}
