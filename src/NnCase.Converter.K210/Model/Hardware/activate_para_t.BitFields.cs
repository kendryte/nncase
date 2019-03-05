using BitFields;

namespace NnCase.Converter.K210.Model.Hardware
{
    partial struct activate_para_t
    {
        public ulong Value;

        private const int shift_numberShift = 0;
        private const ulong shift_numberMask = unchecked((ulong)((1UL << 8) - (1UL << 0)));
        public Bit8 shift_number
        {
            get => (Bit8)((Value & shift_numberMask) >> shift_numberShift);
            set => Value = unchecked((ulong)((Value & ~shift_numberMask) | ((((ulong)value) << shift_numberShift) & shift_numberMask)));
        }
        private const int y_mulShift = 8;
        private const ulong y_mulMask = unchecked((ulong)((1UL << 24) - (1UL << 8)));
        public Bit16 y_mul
        {
            get => (Bit16)((Value & y_mulMask) >> y_mulShift);
            set => Value = unchecked((ulong)((Value & ~y_mulMask) | ((((ulong)value) << y_mulShift) & y_mulMask)));
        }
        private const int x_startShift = 24;
        private const ulong x_startMask = unchecked((ulong)((1UL << 60) - (1UL << 24)));
        public Bit36 x_start
        {
            get => (Bit36)((Value & x_startMask) >> x_startShift);
            set => Value = unchecked((ulong)((Value & ~x_startMask) | ((((ulong)value) << x_startShift) & x_startMask)));
        }
    }
}
