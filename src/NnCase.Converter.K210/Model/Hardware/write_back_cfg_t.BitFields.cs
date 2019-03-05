using BitFields;

namespace NnCase.Converter.K210.Model.Hardware
{
    partial struct write_back_cfg_t
    {
        public ulong Value;

        private const int wb_channel_switch_addrShift = 0;
        private const ulong wb_channel_switch_addrMask = unchecked((ulong)((1UL << 15) - (1UL << 0)));
        public Bit15 wb_channel_switch_addr
        {
            get => (Bit15)((Value & wb_channel_switch_addrMask) >> wb_channel_switch_addrShift);
            set => Value = unchecked((ulong)((Value & ~wb_channel_switch_addrMask) | ((((ulong)value) << wb_channel_switch_addrShift) & wb_channel_switch_addrMask)));
        }
        private const int reserved0Shift = 15;
        private const ulong reserved0Mask = unchecked((ulong)((1UL << 16) - (1UL << 15)));
        public Bit1 reserved0
        {
            get => (Bit1)((Value & reserved0Mask) >> reserved0Shift);
            set => Value = unchecked((ulong)((Value & ~reserved0Mask) | ((((ulong)value) << reserved0Shift) & reserved0Mask)));
        }
        private const int wb_row_switch_addrShift = 16;
        private const ulong wb_row_switch_addrMask = unchecked((ulong)((1UL << 20) - (1UL << 16)));
        public Bit4 wb_row_switch_addr
        {
            get => (Bit4)((Value & wb_row_switch_addrMask) >> wb_row_switch_addrShift);
            set => Value = unchecked((ulong)((Value & ~wb_row_switch_addrMask) | ((((ulong)value) << wb_row_switch_addrShift) & wb_row_switch_addrMask)));
        }
        private const int wb_groupShift = 20;
        private const ulong wb_groupMask = unchecked((ulong)((1UL << 23) - (1UL << 20)));
        public Bit3 wb_group
        {
            get => (Bit3)((Value & wb_groupMask) >> wb_groupShift);
            set => Value = unchecked((ulong)((Value & ~wb_groupMask) | ((((ulong)value) << wb_groupShift) & wb_groupMask)));
        }
        private const int reserved1Shift = 23;
        private const ulong reserved1Mask = unchecked((ulong)((1UL << 64) - (1UL << 23)));
        public Bit41 reserved1
        {
            get => (Bit41)((Value & reserved1Mask) >> reserved1Shift);
            set => Value = unchecked((ulong)((Value & ~reserved1Mask) | ((((ulong)value) << reserved1Shift) & reserved1Mask)));
        }
    }
}
