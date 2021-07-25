#include "yolox.h"
#define INCBIN_STYLE INCBIN_STYLE_SNAKE
#define INCBIN_PREFIX
#include "incbin.h"
#include "stdio.h"
INCBIN(output_float, "/Users/lisa/Documents/nncase/examples/yolox/tmp/yolox_nano_float/dog.bin");
INCBIN(output_quant, "/Users/lisa/Documents/nncase/examples/yolox/tmp/yolox_nano/dog.bin");
void drawboxes(uint32_t x1, uint32_t y1, uint32_t x2, uint32_t y2, uint32_t label, float prob)
{
    printf("%d %d %d %d %d %f \r\n", x1, y1, x2, y2, label, prob);
}

int main(int argc, char const *argv[])
{
    const float *boxes_float = (const float *)output_float_data;
    const float *boxes_quant = (const float *)output_quant_data;

    yolox_init(224, 0.1f, 0.1f, 80);
    printf("------- float ---------\r\n");
    yolox_detect(boxes_float, &drawboxes);

    printf("------- quant ---------\r\n");
    yolox_detect(boxes_quant, &drawboxes);
    return 0;
}