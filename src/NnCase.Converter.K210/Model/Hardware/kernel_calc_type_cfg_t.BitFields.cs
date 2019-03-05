using BitFields;

namespace NnCase.Converter.K210.Model.Hardware
{
    partial struct kernel_calc_type_cfg_t
    {
        public ulong Value;

        private const int channel_switch_addrShift = 0;
        private const ulong channel_switch_addrMask = unchecked((ulong)((1UL << 15) - (1UL << 0)));
        public Bit15 channel_switch_addr
        {
            get => (Bit15)((Value & channel_switch_addrMask) >> channel_switch_addrShift);
            set => Value = unchecked((ulong)((Value & ~channel_switch_addrMask) | ((((ulong)value) << channel_switch_addrShift) & channel_switch_addrMask)));
        }
        private const int reservedShift = 15;
        private const ulong reservedMask = unchecked((ulong)((1UL << 16) - (1UL << 15)));
        public Bit1 reserved
        {
            get => (Bit1)((Value & reservedMask) >> reservedShift);
            set => Value = unchecked((ulong)((Value & ~reservedMask) | ((((ulong)value) << reservedShift) & reservedMask)));
        }
        private const int row_switch_addrShift = 16;
        private const ulong row_switch_addrMask = unchecked((ulong)((1UL << 20) - (1UL << 16)));
        public Bit4 row_switch_addr
        {
            get => (Bit4)((Value & row_switch_addrMask) >> row_switch_addrShift);
            set => Value = unchecked((ulong)((Value & ~row_switch_addrMask) | ((((ulong)value) << row_switch_addrShift) & row_switch_addrMask)));
        }
        private const int coef_sizeShift = 20;
        private const ulong coef_sizeMask = unchecked((ulong)((1UL << 28) - (1UL << 20)));
        public Bit8 coef_size
        {
            get => (Bit8)((Value & coef_sizeMask) >> coef_sizeShift);
            set => Value = unchecked((ulong)((Value & ~coef_sizeMask) | ((((ulong)value) << coef_sizeShift) & coef_sizeMask)));
        }
        private const int coef_groupShift = 28;
        private const ulong coef_groupMask = unchecked((ulong)((1UL << 31) - (1UL << 28)));
        public Bit3 coef_group
        {
            get => (Bit3)((Value & coef_groupMask) >> coef_groupShift);
            set => Value = unchecked((ulong)((Value & ~coef_groupMask) | ((((ulong)value) << coef_groupShift) & coef_groupMask)));
        }
        private const int load_actShift = 31;
        private const ulong load_actMask = unchecked((ulong)((1UL << 32) - (1UL << 31)));
        public Bit1 load_act
        {
            get => (Bit1)((Value & load_actMask) >> load_actShift);
            set => Value = unchecked((ulong)((Value & ~load_actMask) | ((((ulong)value) << load_actShift) & load_actMask)));
        }
        private const int active_addrShift = 32;
        private const ulong active_addrMask = unchecked((ulong)((1UL << 64) - (1UL << 32)));
        public Bit32 active_addr
        {
            get => (Bit32)((Value & active_addrMask) >> active_addrShift);
            set => Value = unchecked((ulong)((Value & ~active_addrMask) | ((((ulong)value) << active_addrShift) & active_addrMask)));
        }
    }
}
