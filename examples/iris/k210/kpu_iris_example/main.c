/* Copyright 2018 Canaan Inc.
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
#include <stdio.h>
#include "kpu.h"
#include <platform.h>
#include <printf.h>
#include <string.h>
#include <stdlib.h>
#include <sysctl.h>
#include <float.h>
#include "uarths.h"
#define INCBIN_STYLE INCBIN_STYLE_SNAKE
#define INCBIN_PREFIX
#include "incbin.h"

#define CLASS10 1

#define PLL0_OUTPUT_FREQ 1000000000UL
#define PLL1_OUTPUT_FREQ 400000000UL
#define PLL2_OUTPUT_FREQ 45158400UL

volatile uint32_t g_ai_done_flag;

const float features[] = {5.1,3.8,1.9,0.4};
const char *labels[] = { "setosa", "versicolor", "virginica" };
kpu_model_context_t task1;

INCBIN(model, "iris.kmodel");

static void ai_done(void* userdata)
{
    g_ai_done_flag = 1;
    
    float *features;
    size_t count;
    kpu_get_output(&task1, 0, (uint8_t **)&features, &count);
    count /= sizeof(float);

    size_t i;
    for (i = 0; i < count; i++)
    {
        if (i % 64 == 0)
            printf("\n");
        printf("%f, ", features[i]);
    }

    printf("\n");
}

size_t argmax(const float *src, size_t count)
{
    float max = FLT_MIN;
    size_t max_id = 0, i;
    for (i = 0; i < count; i++)
    {
        if (src[i] > max)
        {
            max = src[i];
            max_id = i;
        }
    }

    return max_id;
}

int main()
{
    /* Set CPU and dvp clk */
    //sysctl_pll_set_freq(SYSCTL_PLL0, PLL0_OUTPUT_FREQ);
    sysctl_pll_set_freq(SYSCTL_PLL1, PLL1_OUTPUT_FREQ);
    //sysctl_pll_set_freq(SYSCTL_PLL2, PLL2_OUTPUT_FREQ);
    sysctl_clock_enable(SYSCTL_CLOCK_AI);
    uarths_init();
    plic_init();
    sysctl_enable_irq();
    
    if (kpu_load_kmodel(&task1, model_data) != 0)
    {
        printf("Cannot load kmodel.\n");
        exit(-1);
    }
      
    int j;
    for (j = 0; j < 1; j++)
    {
        g_ai_done_flag = 0;

        if (kpu_run_kmodel(&task1, (const uint8_t *)features, 5, ai_done, NULL) != 0)
        {
            printf("Cannot run kmodel.\n");
            exit(-1);
        }
		while (!g_ai_done_flag);

        float *output;
        size_t output_size;
        kpu_get_output(&task1, 0, (uint8_t **)&output, &output_size);
        puts(labels[argmax(output, output_size / 4)]);
    }

    while (1)
        ;    
}