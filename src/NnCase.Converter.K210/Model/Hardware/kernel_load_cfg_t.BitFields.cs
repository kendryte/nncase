using BitFields;

namespace NnCase.Converter.K210.Model.Hardware
{
    partial struct kernel_load_cfg_t
    {
        public ulong Value;

        private const int load_coorShift = 0;
        private const ulong load_coorMask = unchecked((ulong)((1UL << 1) - (1UL << 0)));
        public Bit1 load_coor
        {
            get => (Bit1)((Value & load_coorMask) >> load_coorShift);
            set => Value = unchecked((ulong)((Value & ~load_coorMask) | ((((ulong)value) << load_coorShift) & load_coorMask)));
        }
        private const int load_timeShift = 1;
        private const ulong load_timeMask = unchecked((ulong)((1UL << 7) - (1UL << 1)));
        public Bit6 load_time
        {
            get => (Bit6)((Value & load_timeMask) >> load_timeShift);
            set => Value = unchecked((ulong)((Value & ~load_timeMask) | ((((ulong)value) << load_timeShift) & load_timeMask)));
        }
        private const int reserved0Shift = 7;
        private const ulong reserved0Mask = unchecked((ulong)((1UL << 15) - (1UL << 7)));
        public Bit8 reserved0
        {
            get => (Bit8)((Value & reserved0Mask) >> reserved0Shift);
            set => Value = unchecked((ulong)((Value & ~reserved0Mask) | ((((ulong)value) << reserved0Shift) & reserved0Mask)));
        }
        private const int para_sizeShift = 15;
        private const ulong para_sizeMask = unchecked((ulong)((1UL << 32) - (1UL << 15)));
        public Bit17 para_size
        {
            get => (Bit17)((Value & para_sizeMask) >> para_sizeShift);
            set => Value = unchecked((ulong)((Value & ~para_sizeMask) | ((((ulong)value) << para_sizeShift) & para_sizeMask)));
        }
        private const int para_start_addrShift = 32;
        private const ulong para_start_addrMask = unchecked((ulong)((1UL << 64) - (1UL << 32)));
        public Bit32 para_start_addr
        {
            get => (Bit32)((Value & para_start_addrMask) >> para_start_addrShift);
            set => Value = unchecked((ulong)((Value & ~para_start_addrMask) | ((((ulong)value) << para_start_addrShift) & para_start_addrMask)));
        }
    }
}
