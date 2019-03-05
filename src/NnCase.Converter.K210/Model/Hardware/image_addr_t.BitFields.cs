using BitFields;

namespace NnCase.Converter.K210.Model.Hardware
{
    partial struct image_addr_t
    {
        public ulong Value;

        private const int image_src_addrShift = 0;
        private const ulong image_src_addrMask = unchecked((ulong)((1UL << 15) - (1UL << 0)));
        public Bit15 image_src_addr
        {
            get => (Bit15)((Value & image_src_addrMask) >> image_src_addrShift);
            set => Value = unchecked((ulong)((Value & ~image_src_addrMask) | ((((ulong)value) << image_src_addrShift) & image_src_addrMask)));
        }
        private const int reserved0Shift = 15;
        private const ulong reserved0Mask = unchecked((ulong)((1UL << 32) - (1UL << 15)));
        public Bit17 reserved0
        {
            get => (Bit17)((Value & reserved0Mask) >> reserved0Shift);
            set => Value = unchecked((ulong)((Value & ~reserved0Mask) | ((((ulong)value) << reserved0Shift) & reserved0Mask)));
        }
        private const int image_dst_addrShift = 32;
        private const ulong image_dst_addrMask = unchecked((ulong)((1UL << 47) - (1UL << 32)));
        public Bit15 image_dst_addr
        {
            get => (Bit15)((Value & image_dst_addrMask) >> image_dst_addrShift);
            set => Value = unchecked((ulong)((Value & ~image_dst_addrMask) | ((((ulong)value) << image_dst_addrShift) & image_dst_addrMask)));
        }
        private const int reserved1Shift = 47;
        private const ulong reserved1Mask = unchecked((ulong)((1UL << 64) - (1UL << 47)));
        public Bit17 reserved1
        {
            get => (Bit17)((Value & reserved1Mask) >> reserved1Shift);
            set => Value = unchecked((ulong)((Value & ~reserved1Mask) | ((((ulong)value) << reserved1Shift) & reserved1Mask)));
        }
    }
}
