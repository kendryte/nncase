// #include "pred.h"
// #include "prior.h"
// #include "region_layer.h"
// #include <stdio.h>

// int main(int argc, char const *argv[]) {
//     region_layer_t rl;
//     box_info_t bx;
//     float variances[2]= {0.1, 0.2};
//     region_layer_init(&rl, anchor, 3160, 4, 5, 1, 320, 240, 0.7, 0.4, variances);
//     boxes_info_init(&rl, &bx, 200);
//     rl.clses_input= pred_conf;
//     rl.bbox_input= pred_bbox;
//     rl.landm_input= pred_landm;
//     region_layer_run(&rl, &bx);

//     region_layer_draw_boxes(&bx, NULL);
//     boxes_info_reset(&bx);
//     return 0;
// }
