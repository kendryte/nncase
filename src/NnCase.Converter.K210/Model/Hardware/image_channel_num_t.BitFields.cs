using BitFields;

namespace NnCase.Converter.K210.Model.Hardware
{
    partial struct image_channel_num_t
    {
        public ulong Value;

        private const int i_ch_numShift = 0;
        private const ulong i_ch_numMask = unchecked((ulong)((1UL << 10) - (1UL << 0)));
        public Bit10 i_ch_num
        {
            get => (Bit10)((Value & i_ch_numMask) >> i_ch_numShift);
            set => Value = unchecked((ulong)((Value & ~i_ch_numMask) | ((((ulong)value) << i_ch_numShift) & i_ch_numMask)));
        }
        private const int reserved0Shift = 10;
        private const ulong reserved0Mask = unchecked((ulong)((1UL << 32) - (1UL << 10)));
        public Bit22 reserved0
        {
            get => (Bit22)((Value & reserved0Mask) >> reserved0Shift);
            set => Value = unchecked((ulong)((Value & ~reserved0Mask) | ((((ulong)value) << reserved0Shift) & reserved0Mask)));
        }
        private const int o_ch_numShift = 32;
        private const ulong o_ch_numMask = unchecked((ulong)((1UL << 42) - (1UL << 32)));
        public Bit10 o_ch_num
        {
            get => (Bit10)((Value & o_ch_numMask) >> o_ch_numShift);
            set => Value = unchecked((ulong)((Value & ~o_ch_numMask) | ((((ulong)value) << o_ch_numShift) & o_ch_numMask)));
        }
        private const int reserved1Shift = 42;
        private const ulong reserved1Mask = unchecked((ulong)((1UL << 48) - (1UL << 42)));
        public Bit6 reserved1
        {
            get => (Bit6)((Value & reserved1Mask) >> reserved1Shift);
            set => Value = unchecked((ulong)((Value & ~reserved1Mask) | ((((ulong)value) << reserved1Shift) & reserved1Mask)));
        }
        private const int o_ch_num_coefShift = 48;
        private const ulong o_ch_num_coefMask = unchecked((ulong)((1UL << 58) - (1UL << 48)));
        public Bit10 o_ch_num_coef
        {
            get => (Bit10)((Value & o_ch_num_coefMask) >> o_ch_num_coefShift);
            set => Value = unchecked((ulong)((Value & ~o_ch_num_coefMask) | ((((ulong)value) << o_ch_num_coefShift) & o_ch_num_coefMask)));
        }
        private const int reserved2Shift = 58;
        private const ulong reserved2Mask = unchecked((ulong)((1UL << 64) - (1UL << 58)));
        public Bit6 reserved2
        {
            get => (Bit6)((Value & reserved2Mask) >> reserved2Shift);
            set => Value = unchecked((ulong)((Value & ~reserved2Mask) | ((((ulong)value) << reserved2Shift) & reserved2Mask)));
        }
    }
}
