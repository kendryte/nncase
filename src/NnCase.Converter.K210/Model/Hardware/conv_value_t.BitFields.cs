using BitFields;

namespace NnCase.Converter.K210.Model.Hardware
{
    partial struct conv_value_t
    {
        public ulong Value;

        private const int shr_wShift = 0;
        private const ulong shr_wMask = unchecked((ulong)((1UL << 4) - (1UL << 0)));
        public Bit4 shr_w
        {
            get => (Bit4)((Value & shr_wMask) >> shr_wShift);
            set => Value = unchecked((ulong)((Value & ~shr_wMask) | ((((ulong)value) << shr_wShift) & shr_wMask)));
        }
        private const int shr_xShift = 4;
        private const ulong shr_xMask = unchecked((ulong)((1UL << 8) - (1UL << 4)));
        public Bit4 shr_x
        {
            get => (Bit4)((Value & shr_xMask) >> shr_xShift);
            set => Value = unchecked((ulong)((Value & ~shr_xMask) | ((((ulong)value) << shr_xShift) & shr_xMask)));
        }
        private const int arg_wShift = 8;
        private const ulong arg_wMask = unchecked((ulong)((1UL << 32) - (1UL << 8)));
        public Bit24 arg_w
        {
            get => (Bit24)((Value & arg_wMask) >> arg_wShift);
            set => Value = unchecked((ulong)((Value & ~arg_wMask) | ((((ulong)value) << arg_wShift) & arg_wMask)));
        }
        private const int arg_xShift = 32;
        private const ulong arg_xMask = unchecked((ulong)((1UL << 56) - (1UL << 32)));
        public Bit24 arg_x
        {
            get => (Bit24)((Value & arg_xMask) >> arg_xShift);
            set => Value = unchecked((ulong)((Value & ~arg_xMask) | ((((ulong)value) << arg_xShift) & arg_xMask)));
        }
        private const int reserved0Shift = 56;
        private const ulong reserved0Mask = unchecked((ulong)((1UL << 64) - (1UL << 56)));
        public Bit8 reserved0
        {
            get => (Bit8)((Value & reserved0Mask) >> reserved0Shift);
            set => Value = unchecked((ulong)((Value & ~reserved0Mask) | ((((ulong)value) << reserved0Shift) & reserved0Mask)));
        }
    }
}
