/*
 * vad.h
 *
 *  Created on: 2025年7月21日
 *      Author: 廖先强
 */

#ifndef GALAXY_SDK_VAD_H_
#define GALAXY_SDK_VAD_H_

#include <model_parameters_vad.h>
#include "math.h"
#include "float.h"
#include "constants.h"
#include "conv.h"
#include "model_parameters_speaker.h"
#include "string.h"
#include "riscv_math.h"
#include <stdio.h>
#include "precomputed_meldata.h"  //cyw7


typedef struct {
    float real;
    float imag;
} Complex;

int vad_preprocess(float *inp_data, Conv2dData *out_data);
//int vad(Conv2dData *inp_data, float *speech_probability);
int initialize_mel_resources();
int speaker_classifier_inference(Conv2dData *inp_data, float probabilities[5]);
void free_mel_resources();
int residual_block_forward(
    Conv2dData *input_feat,
    Conv2dData *output_feat,

    const float *conv1_w, BatchNorm2d *bn1_params,
    const float *conv2_w, BatchNorm2d *bn2_params,
    uint16_t main_out_channels,
    bool has_shortcut_conv,
    const float *conv_s_w, BatchNorm2d *bn_s_params,

    float *temp_main_path_buf1,
    float *temp_main_path_buf2,
    float *temp_shortcut_path_buf,
    float *temp_padded_buf
) ;
int vad_inference(Conv2dData *inp_data, float *speech_probability);
int speaker_classifier_inference(Conv2dData *inp_data, float probabilities[5]);

extern float g_hann_coeffs[512];
void initialize_hann_window_coeffs(void);
void fft_module_init(int fftLen);


#endif /* GALAXY_SDK_VAD_H_ */
