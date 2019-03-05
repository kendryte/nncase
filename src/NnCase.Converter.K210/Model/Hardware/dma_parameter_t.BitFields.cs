using BitFields;

namespace NnCase.Converter.K210.Model.Hardware
{
    partial struct dma_parameter_t
    {
        public ulong Value;

        private const int send_data_outShift = 0;
        private const ulong send_data_outMask = unchecked((ulong)((1UL << 1) - (1UL << 0)));
        public Bit1 send_data_out
        {
            get => (Bit1)((Value & send_data_outMask) >> send_data_outShift);
            set => Value = unchecked((ulong)((Value & ~send_data_outMask) | ((((ulong)value) << send_data_outShift) & send_data_outMask)));
        }
        private const int reservedShift = 1;
        private const ulong reservedMask = unchecked((ulong)((1UL << 16) - (1UL << 1)));
        public Bit15 reserved
        {
            get => (Bit15)((Value & reservedMask) >> reservedShift);
            set => Value = unchecked((ulong)((Value & ~reservedMask) | ((((ulong)value) << reservedShift) & reservedMask)));
        }
        private const int channel_byte_numShift = 16;
        private const ulong channel_byte_numMask = unchecked((ulong)((1UL << 32) - (1UL << 16)));
        public Bit16 channel_byte_num
        {
            get => (Bit16)((Value & channel_byte_numMask) >> channel_byte_numShift);
            set => Value = unchecked((ulong)((Value & ~channel_byte_numMask) | ((((ulong)value) << channel_byte_numShift) & channel_byte_numMask)));
        }
        private const int dma_total_byteShift = 32;
        private const ulong dma_total_byteMask = unchecked((ulong)((1UL << 64) - (1UL << 32)));
        public Bit32 dma_total_byte
        {
            get => (Bit32)((Value & dma_total_byteMask) >> dma_total_byteShift);
            set => Value = unchecked((ulong)((Value & ~dma_total_byteMask) | ((((ulong)value) << dma_total_byteShift) & dma_total_byteMask)));
        }
    }
}
