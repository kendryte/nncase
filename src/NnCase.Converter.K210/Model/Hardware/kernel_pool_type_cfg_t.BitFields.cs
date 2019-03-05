using BitFields;

namespace NnCase.Converter.K210.Model.Hardware
{
    partial struct kernel_pool_type_cfg_t
    {
        public ulong Value;

        private const int kernel_typeShift = 0;
        private const ulong kernel_typeMask = unchecked((ulong)((1UL << 3) - (1UL << 0)));
        public Bit3 kernel_type
        {
            get => (Bit3)((Value & kernel_typeMask) >> kernel_typeShift);
            set => Value = unchecked((ulong)((Value & ~kernel_typeMask) | ((((ulong)value) << kernel_typeShift) & kernel_typeMask)));
        }
        private const int pad_typeShift = 3;
        private const ulong pad_typeMask = unchecked((ulong)((1UL << 4) - (1UL << 3)));
        public Bit1 pad_type
        {
            get => (Bit1)((Value & pad_typeMask) >> pad_typeShift);
            set => Value = unchecked((ulong)((Value & ~pad_typeMask) | ((((ulong)value) << pad_typeShift) & pad_typeMask)));
        }
        private const int pool_typeShift = 4;
        private const ulong pool_typeMask = unchecked((ulong)((1UL << 8) - (1UL << 4)));
        public Bit4 pool_type
        {
            get => (Bit4)((Value & pool_typeMask) >> pool_typeShift);
            set => Value = unchecked((ulong)((Value & ~pool_typeMask) | ((((ulong)value) << pool_typeShift) & pool_typeMask)));
        }
        private const int first_strideShift = 8;
        private const ulong first_strideMask = unchecked((ulong)((1UL << 9) - (1UL << 8)));
        public Bit1 first_stride
        {
            get => (Bit1)((Value & first_strideMask) >> first_strideShift);
            set => Value = unchecked((ulong)((Value & ~first_strideMask) | ((((ulong)value) << first_strideShift) & first_strideMask)));
        }
        private const int bypass_convShift = 9;
        private const ulong bypass_convMask = unchecked((ulong)((1UL << 10) - (1UL << 9)));
        public Bit1 bypass_conv
        {
            get => (Bit1)((Value & bypass_convMask) >> bypass_convShift);
            set => Value = unchecked((ulong)((Value & ~bypass_convMask) | ((((ulong)value) << bypass_convShift) & bypass_convMask)));
        }
        private const int load_paraShift = 10;
        private const ulong load_paraMask = unchecked((ulong)((1UL << 11) - (1UL << 10)));
        public Bit1 load_para
        {
            get => (Bit1)((Value & load_paraMask) >> load_paraShift);
            set => Value = unchecked((ulong)((Value & ~load_paraMask) | ((((ulong)value) << load_paraShift) & load_paraMask)));
        }
        private const int reserved0Shift = 11;
        private const ulong reserved0Mask = unchecked((ulong)((1UL << 16) - (1UL << 11)));
        public Bit5 reserved0
        {
            get => (Bit5)((Value & reserved0Mask) >> reserved0Shift);
            set => Value = unchecked((ulong)((Value & ~reserved0Mask) | ((((ulong)value) << reserved0Shift) & reserved0Mask)));
        }
        private const int dma_burst_sizeShift = 16;
        private const ulong dma_burst_sizeMask = unchecked((ulong)((1UL << 24) - (1UL << 16)));
        public Bit8 dma_burst_size
        {
            get => (Bit8)((Value & dma_burst_sizeMask) >> dma_burst_sizeShift);
            set => Value = unchecked((ulong)((Value & ~dma_burst_sizeMask) | ((((ulong)value) << dma_burst_sizeShift) & dma_burst_sizeMask)));
        }
        private const int pad_valueShift = 24;
        private const ulong pad_valueMask = unchecked((ulong)((1UL << 32) - (1UL << 24)));
        public Bit8 pad_value
        {
            get => (Bit8)((Value & pad_valueMask) >> pad_valueShift);
            set => Value = unchecked((ulong)((Value & ~pad_valueMask) | ((((ulong)value) << pad_valueShift) & pad_valueMask)));
        }
        private const int bwsx_base_addrShift = 32;
        private const ulong bwsx_base_addrMask = unchecked((ulong)((1UL << 64) - (1UL << 32)));
        public Bit32 bwsx_base_addr
        {
            get => (Bit32)((Value & bwsx_base_addrMask) >> bwsx_base_addrShift);
            set => Value = unchecked((ulong)((Value & ~bwsx_base_addrMask) | ((((ulong)value) << bwsx_base_addrShift) & bwsx_base_addrMask)));
        }
    }
}
