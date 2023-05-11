#include "op_executor.h"
#include "tensor.h"

// ops which be Not suitable for auto-generated
ORTKI_API(ortki::OrtKITensorSeq) * ortki_Split(ortki::OrtKITensor * input, ortki::OrtKITensor * split, int64_t axis);

// only one of size and scale can be passed
ORTKI_API(ortki::OrtKITensor *) ortki_ResizeWithSizes(ortki::OrtKITensor * X, ortki::OrtKITensor * roi, ortki::OrtKITensor * sizes, const char* coordinate_transformation_mode, float cubic_coeff_a, int64_t exclude_outside, float extrapolation_value, const char* mode, const char* nearest_mode);

ORTKI_API(ortki::OrtKITensor *) ortki_ResizeWithScales(ortki::OrtKITensor * X, ortki::OrtKITensor * roi, ortki::OrtKITensor * scales, const char* coordinate_transformation_mode, float cubic_coeff_a, int64_t exclude_outside, float extrapolation_value, const char* mode, const char* nearest_mode);

// Upsample has been deprecated
ORTKI_API(ortki::OrtKITensor *) ortki_Upsample(ortki::OrtKITensor * X, ortki::OrtKITensor * scales, const char* mode);

// training mode set false and must spec only one output
ORTKI_API(ortki::OrtKITensor *) ortki_BatchNormalization(ortki::OrtKITensor * X, ortki::OrtKITensor * scale, ortki::OrtKITensor * B, ortki::OrtKITensor * input_mean, ortki::OrtKITensor * input_var, float epsilon, float momentum);

// LSTM clip can be not
ORTKI_API(ortki::OrtKITensorSeq *) ortki_LSTM(ortki::OrtKITensor * X, ortki::OrtKITensor * W, ortki::OrtKITensor * R, ortki::OrtKITensor * B, ortki::OrtKITensor * sequence_lens, ortki::OrtKITensor * initial_h, ortki::OrtKITensor * initial_c, ortki::OrtKITensor * P, float* activation_alpha, int activation_alpha_size, float* activation_beta, int activation_beta_size, const char** activations, int activations_size, float clip, const char* direction, int64_t hidden_size, int64_t input_forget, int64_t layout, bool has_clip, int64_t output_size);
