/* Copyright 2021 Canaan Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "board_config.h"
#include "fpioa.h"
#include "gpiohs.h"
#include "image_process.h"
#include "kpu.h"
#include "lcd.h"
#include "nt35310.h"
#include "plic.h"
#include "sysctl.h"
#include "uarths.h"
#include "utils.h"
#define INCBIN_STYLE INCBIN_STYLE_SNAKE
#define INCBIN_PREFIX
#include "incbin.h"
#include "iomem.h"
#include "yolox.h"

#define PLL0_OUTPUT_FREQ 800000000UL
#define PLL1_OUTPUT_FREQ 400000000UL

#define FRAME_WIDTH 224
#define FRAME_HEIGHT 224
#define NUM_BUFFERS 1

volatile uint32_t g_ai_done_flag;
static image_t kpu_image[NUM_BUFFERS];
static uint32_t lcd_gram[FRAME_WIDTH * FRAME_HEIGHT] __attribute__((aligned(32)));

kpu_model_context_t task;

INCBIN(model, "yolox_nano_224.kmodel");

INCBIN(input, "input.bin");

void set_kpu_addr_comps(image_t *image);

static void ai_done(void *ctx)
{
    g_ai_done_flag = 1;
}

#if BOARD_LICHEEDAN
static void io_mux_init(void)
{
    /* Init DVP IO map and function settings */
    fpioa_set_function(42, FUNC_CMOS_RST);
    fpioa_set_function(44, FUNC_CMOS_PWDN);
    fpioa_set_function(46, FUNC_CMOS_XCLK);
    fpioa_set_function(43, FUNC_CMOS_VSYNC);
    fpioa_set_function(45, FUNC_CMOS_HREF);
    fpioa_set_function(47, FUNC_CMOS_PCLK);
    fpioa_set_function(41, FUNC_SCCB_SCLK);
    fpioa_set_function(40, FUNC_SCCB_SDA);

    /* Init SPI IO map and function settings */
    fpioa_set_function(38, FUNC_GPIOHS0 + DCX_GPIONUM);
    fpioa_set_function(36, FUNC_SPI0_SS3);
    fpioa_set_function(39, FUNC_SPI0_SCLK);
    fpioa_set_function(37, FUNC_GPIOHS0 + RST_GPIONUM);

    sysctl_set_spi0_dvp_data(1);
}

static void io_set_power(void)
{
    /* Set dvp and spi pin to 1.8V */
    sysctl_set_power_mode(SYSCTL_POWER_BANK6, SYSCTL_POWER_V18);
    sysctl_set_power_mode(SYSCTL_POWER_BANK7, SYSCTL_POWER_V18);
}

#else
static void io_mux_init(void)
{
    /* Init DVP IO map and function settings */
    fpioa_set_function(11, FUNC_CMOS_RST);
    fpioa_set_function(13, FUNC_CMOS_PWDN);
    fpioa_set_function(14, FUNC_CMOS_XCLK);
    fpioa_set_function(12, FUNC_CMOS_VSYNC);
    fpioa_set_function(17, FUNC_CMOS_HREF);
    fpioa_set_function(15, FUNC_CMOS_PCLK);
    fpioa_set_function(10, FUNC_SCCB_SCLK);
    fpioa_set_function(9, FUNC_SCCB_SDA);

    /* Init SPI IO map and function settings */
    fpioa_set_function(8, FUNC_GPIOHS0 + DCX_GPIONUM);
    fpioa_set_function(6, FUNC_SPI0_SS3);
    fpioa_set_function(7, FUNC_SPI0_SCLK);

    sysctl_set_spi0_dvp_data(1);
}

static void io_set_power(void)
{
    /* Set dvp and spi pin to 1.8V */
    sysctl_set_power_mode(SYSCTL_POWER_BANK1, SYSCTL_POWER_V18);
    sysctl_set_power_mode(SYSCTL_POWER_BANK2, SYSCTL_POWER_V18);
}
#endif

static void drawboxes(uint32_t x1, uint32_t y1, uint32_t x2, uint32_t y2, uint32_t label, float prob)
{
    if(x1 >= 320)
        x1 = 319;
    if(x2 >= 320)
        x2 = 319;
    if(y1 >= 240)
        y1 = 239;
    if(y2 >= 240)
        y2 = 239;

    printf("%d %d %d %d %d %f \r\n", x1, y1, x2, y2, label, prob);
    lcd_draw_rectangle(x1, y1, x2, y2, 1, RED);
}

void rgb888_to_lcd(uint8_t *src, uint16_t *dest, size_t width, size_t height)
{
    size_t i, chn_size = width * height;
    for(size_t i = 0; i < width * height; i++)
    {
        uint8_t r = src[i];
        uint8_t g = src[chn_size + i];
        uint8_t b = src[chn_size * 2 + i];

        uint16_t rgb = ((r & 0b11111000) << 8) | ((g & 0b11111100) << 3) | (b >> 3);
        size_t d_i = i % 2 ? (i - 1) : (i + 1);
        dest[d_i] = rgb;
    }
}

int main(void)
{
    /* Set CPU and dvp clk */
    /* Set CPU and dvp clk */
    sysctl_pll_set_freq(SYSCTL_PLL0, PLL0_OUTPUT_FREQ);
    sysctl_pll_set_freq(SYSCTL_PLL1, PLL1_OUTPUT_FREQ);
    uarths_init();

    sysctl_clock_enable(SYSCTL_CLOCK_AI);
    uarths_init();
    uarths_config(115200, 1);

    io_mux_init();
    io_set_power();
    plic_init();
    /* LCD init */
    printf("LCD init\n");
    lcd_init();
#if BOARD_LICHEEDAN
    lcd_set_direction(DIR_YX_RLDU);
#else
    lcd_set_direction(DIR_YX_RLUD);
#endif

    uint8_t *model_data_align = model_data;
    kpu_image[0].pixel = 1;
    kpu_image[0].width = FRAME_WIDTH;
    kpu_image[0].height = FRAME_HEIGHT;
    image_init(&kpu_image[0]);
    memcpy(kpu_image[0].addr, input_data, input_size);
    printf("input data uploaded\r\n");

    /* init face detect model */
    if(kpu_load_kmodel(&task, model_data_align) != 0)
    {
        printf("\r\nmodel init error\r\n");
        while(1)
            ;
    }
    printf("yolox nano init success!\r\n");

    /* enable global interrupt */
    sysctl_enable_irq();

    /* system start */
    printf("System start\r\n");

    yolox_init(224, 0.1f, 0.1f, 80);
    printf("yolox init\r\n");

    while(1)
    {
        g_ai_done_flag = 0;
        /* tic */
        uint64_t kpu_start_time = sysctl_get_time_us();

        kpu_run_kmodel(&task, kpu_image[0].addr, DMAC_CHANNEL5, ai_done, NULL);

        while(!g_ai_done_flag)
            ;

        uint64_t kpu_end_time = sysctl_get_time_us();

        float *boxes;
        size_t output_size;
        kpu_get_output(&task, 0, (uint8_t **)&boxes, &output_size);

        /* display pic*/
        rgb888_to_lcd(kpu_image[0].addr, lcd_gram, 224, 224);
        lcd_draw_picture(0, 0, 224, 224, lcd_gram);

        /* draw boxs */
        yolox_detect(boxes, &drawboxes);

        uint64_t forward_end_time = sysctl_get_time_us();

        /* toc */
        printf("model: %.03f ms , postprocess: %.03f ms \r\n", (kpu_end_time - kpu_start_time) / 1000.0, (forward_end_time - kpu_end_time) / 1000.0);
    }
}
