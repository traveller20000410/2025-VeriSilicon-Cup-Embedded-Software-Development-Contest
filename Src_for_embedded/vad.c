/*
 * vad.c
 *
 *  Created on: 2025年7月21日
 *      Author: 廖先强,柴焰旺
 */


#include "vad.h"
#include "osal.h"
#include "riscv_math.h"

/* --- Shared temporary buffers for model inference --- */
// Size is determined by the maximum requirement between VAD and Speaker Classifier models.
#define MODEL_MAX_CONV_BUF_SIZE         (4864)
#define MODEL_MAX_POOL_BUF_SIZE         (1152)
#define MODEL_MAX_PADDED_BUF_SIZE       (5712)
#define MODEL_MAX_RES_MAIN_BUF_SIZE     (4864)
#define MODEL_MAX_RES_SHORTCUT_BUF_SIZE (4864)
#define MODEL_MAX_ADAPTIVE_POOL_VEC_SIZE (24)

static float model_res_main_path_buf[MODEL_MAX_RES_MAIN_BUF_SIZE];
static float model_res_shortcut_path_buf[MODEL_MAX_RES_SHORTCUT_BUF_SIZE];
static float model_temp_conv_buf[MODEL_MAX_CONV_BUF_SIZE];
static float model_temp_pool_buf[MODEL_MAX_POOL_BUF_SIZE];
static float model_res_padded_buf[MODEL_MAX_PADDED_BUF_SIZE];
static float model_temp_adaptive_pool_out_buf[MODEL_MAX_ADAPTIVE_POOL_VEC_SIZE];
//static float model_res_main_path_buf[MODEL_MAX_RES_MAIN_BUF_SIZE];
//static float model_res_shortcut_path_buf[MODEL_MAX_RES_SHORTCUT_BUF_SIZE];

static riscv_rfft_fast_instance_f32 S_RFFT;

// 在系统启动或第一次调用前做一次初始化
void fft_module_init(int fftLen) {
    if (riscv_rfft_fast_init_f32(&S_RFFT, fftLen) != RISCV_MATH_SUCCESS) {
        printf("RFFT init failed for length %d\n", fftLen);
    }
}

static void padding_value(const Conv2dData *raw_data, uint16_t pad_len,
                         float pad_value,float *paded_data)
{
    uint16_t row = raw_data->row, col = raw_data->col, chan = raw_data->channel;
    uint16_t paded_data_size = 0;
    uint16_t i = 0, j = 0, k = 0;
    uint16_t pad_idx = 0;

    paded_data_size = (row + 2 * pad_len) * (col + 2 * pad_len) * chan;

    for (i = 0; i < paded_data_size; i++) {
        paded_data[i] = pad_value;
    }

    for (i = 0; i < raw_data->channel; i++) {
        for (j = 0; j < raw_data->row; j++) {
            for (k = 0; k < raw_data->col; k++) {
                pad_idx = k + pad_len + (j + pad_len) * (col + 2 * pad_len) +
                    i * (row + 2 * pad_len) * (col + 2 * pad_len);
                paded_data[pad_idx] = raw_data->data[k + j * col + i * row * col];
            }
        }
    }
}

void fft_with_nmsis(const float *inp, float *out, int fftLen) {
    riscv_rfft_fast_f32(&S_RFFT, (float32_t*)inp, (float32_t*)out, 0);
}


//static float **g_mel_filterbank = NULL;
//// 静态分配梅尔滤波器组的存储空间
//static float g_mel_filterbank_data[N_MELS][N_FFT_MEL / 2 + 1];  //cyw7

// 仍然使用一个指针数组，以保持 `create_mel_filterbank` 接口的兼容性
static float *g_mel_filterbank[N_MELS];

//static Complex *g_fft_buffer = NULL;
//static float *g_power_spectrum_buffer = NULL;
//static float *g_fft_window_buffer = NULL;
// 静态分配其他所有缓冲区
//static Complex g_fft_buffer[N_FFT_MEL];
static float   g_fft_output_buffer[N_FFT_MEL];  //cyw8
static float   g_power_spectrum_buffer[N_FFT_MEL / 2 + 1];
static float   g_fft_window_buffer[N_FFT_MEL];

float g_hann_coeffs[512];
static int hann_coeffs_initialized = 0;
// 初始化函数
void initialize_hann_window_coeffs(void) {
    if (!hann_coeffs_initialized) {
        riscv_hanning_f32(g_hann_coeffs, (uint32_t)512);
        hann_coeffs_initialized = 1;
    }
}

static void apply_hann_window_sdk_optimized(const float *input, float *output, uint32_t window_len) {
    if (window_len == 0) {
        return;
    }

    riscv_mult_f32(input, g_hann_coeffs, output, window_len);
}

// 优化后的初始化函数  cyw7
int initialize_mel_resources() {
	static int mel_resources_initialized = 0;
	if (mel_resources_initialized) {
	    return ALGO_NORMAL;
	}

	// —— 从 DLM 堆动态分配 ——

//	printf("Attempting to allocate model buffers from DLM heap...\n");
//
//	// 1. 分配主路径缓冲区
//	model_res_main_path_buf = (float *)osal_malloc(
//	    MODEL_MAX_RES_MAIN_BUF_SIZE * sizeof(float)
//	);
//	if (!model_res_main_path_buf) {
//	    printf("CRITICAL: Failed to allocate model_res_main_path_buf! (Size: %u bytes)\n",
//	      MODEL_MAX_RES_MAIN_BUF_SIZE * sizeof(float)
//	    );
//	    return ALGO_MALLOC_FAIL;
//	}
//	printf(" -> model_res_main_path_buf allocated at %p\n",
//	       (void *)model_res_main_path_buf);

//	// 2. 分配快捷路径缓冲区
//	model_res_shortcut_path_buf = (float *)osal_malloc(
//	    MODEL_MAX_RES_SHORTCUT_BUF_SIZE * sizeof(float)
//	);
//	if (!model_res_shortcut_path_buf) {
//	    printf("CRITICAL: Failed to allocate model_res_shortcut_path_buf! (Size: %u bytes)\n",
//	        MODEL_MAX_RES_SHORTCUT_BUF_SIZE * sizeof(float)
//	    );
//	    // 回收之前成功的分配
//	    osal_free(model_res_main_path_buf);
//	    model_res_main_path_buf = NULL;
//	    return ALGO_MALLOC_FAIL;
//	}
//	printf(" -> model_res_shortcut_path_buf allocated at %p\n",
//	       (void *)model_res_shortcut_path_buf);
	// —— 动态分配结束 ——

	// 将指针数组的每个元素指向Flash中常量二维数组的每一行
	// 需要进行类型转换 (float*) 是因为 g_mel_filterbank_data 是 const
	for (int i = 0; i < N_MELS; i++) {
	    g_mel_filterbank[i] = (float*)g_mel_filterbank_data[i];
	}

	mel_resources_initialized = 1;
	printf("Mel resources initialized from pre-computed Flash data.\n");
	return ALGO_NORMAL;
}

void compute_mel_spectrogram_frame_impl(
    float *inp_audio_chunk,         // 输入音频块,
    int inp_chunk_len,               // inp_audio_chunk 的长度
    float *out_mel_spectrogram,     // 输出梅尔频谱图数组, 大小: N_MELS * NUM_MEL_FRAMES
    int n_fft,                       // N_FFT_MEL (e.g., 512), FFT点数
    int hop_length,                  // HOP_LENGTH_MEL (160)
    int n_mels                      // N_MELS
) {
    if (inp_chunk_len < n_fft) {
        for(int i=0; i < n_mels * NUM_MEL_FRAMES; ++i) out_mel_spectrogram[i] = 0.0F;
        return;
    }

    int num_calculated_mel_frames = (inp_chunk_len - win_length) / hop_length + 1;
    if (num_calculated_mel_frames <= 0) {
        for(int i=0; i < n_mels * NUM_MEL_FRAMES; ++i) out_mel_spectrogram[i] = 0.0F;
        return;
    }

    if (num_calculated_mel_frames > NUM_MEL_FRAMES) {
        num_calculated_mel_frames = NUM_MEL_FRAMES;
    }

    memset(g_fft_window_buffer, 0, sizeof(g_fft_window_buffer));

    for (int frame_idx = 0; frame_idx < num_calculated_mel_frames; frame_idx++) {

        float* current_frame_ptr = inp_audio_chunk + frame_idx * hop_length;

        apply_hann_window_sdk_optimized(current_frame_ptr, g_fft_window_buffer, win_length);

        fft_with_nmsis(g_fft_window_buffer, g_fft_output_buffer, n_fft);

        riscv_cmplx_mag_squared_f32(g_fft_output_buffer,g_power_spectrum_buffer,n_fft / 2 + 1 );

//        for (int i = 0; i < n_fft / 2 + 1; i++) {
//            g_power_spectrum_buffer[i] = g_fft_buffer[i].real * g_fft_buffer[i].real +
//                                          g_fft_buffer[i].imag * g_fft_buffer[i].imag;
//        }

        for (int mel_idx = 0; mel_idx < n_mels; mel_idx++) {
            float mel_energy = 0.0F;
            for (int bin_idx = 0; bin_idx < n_fft / 2 + 1; bin_idx++) {
                mel_energy += g_power_spectrum_buffer[bin_idx] * g_mel_filterbank[mel_idx][bin_idx];
            }

            out_mel_spectrogram[mel_idx * num_calculated_mel_frames + frame_idx] = mel_energy;
        }
    }

    if (num_calculated_mel_frames < NUM_MEL_FRAMES) {
        for (int mel_idx = 0; mel_idx < n_mels; ++mel_idx) {
            for (int frame_idx = num_calculated_mel_frames; frame_idx < NUM_MEL_FRAMES; ++frame_idx) {
                out_mel_spectrogram[mel_idx * NUM_MEL_FRAMES + frame_idx] = 0.0F;
            }
        }
    }

    int total_mel_values = n_mels * NUM_MEL_FRAMES;
    for (int i = 0; i < total_mel_values; i++) {
        out_mel_spectrogram[i] = 10.0F * log10f(out_mel_spectrogram[i] + MEL_EPS);
    }

    //              实现 T.AmplitudeToDB 中 top_db=80 的功能 //111
    // 1. 找到频谱中的最大值
    float max_db = -FLT_MAX; // 使用 float.h 中的最小值初始化
    for (int i = 0; i < total_mel_values; i++) {
        if (out_mel_spectrogram[i] > max_db) {
            max_db = out_mel_spectrogram[i];
        }
    }

    // 2. 计算截断的下限值 (floor)
    const float top_db_val = 80.0f;
    float floor_db = max_db - top_db_val;

    // 3. 将所有低于下限值的数据截断 (clip)
    for (int i = 0; i < total_mel_values; i++) {
        out_mel_spectrogram[i] = fmaxf(out_mel_spectrogram[i], floor_db);
    } //111

    double sum = 0.0F;
    for (int i = 0; i < total_mel_values; i++) {
        sum += out_mel_spectrogram[i];
    }
    float mean = sum / total_mel_values;

    float sum_sq_diff = 0.0;
    for (int i = 0; i < total_mel_values; i++) {
        sum_sq_diff += (out_mel_spectrogram[i] - mean) * (out_mel_spectrogram[i] - mean);
    }

    float std_dev = sqrtf(sum_sq_diff / total_mel_values + MEL_EPS);

    for (int i = 0; i < total_mel_values; i++) {
        if (std_dev > 1e-6F) { // 检查标准差是否过小
            out_mel_spectrogram[i] = (out_mel_spectrogram[i] - mean) / std_dev;
        } else {
            out_mel_spectrogram[i] = out_mel_spectrogram[i] - mean; // 仅中心化
        }
    }
}

int vad_preprocess(float *inp_data, Conv2dData *out_data)
{

    if (!inp_data || !out_data || !out_data->data) {
        return ALGO_POINTER_NULL;
    }

    // 确保梅尔资源已初始化 (现在这个函数开销很小，只是检查一个标志位)
    if (initialize_mel_resources() != ALGO_NORMAL) {
        printf("CRITICAL: Failed to initialize Mel resources.\n");
        return ALGO_ERR_GENERIC;
    }

    compute_mel_spectrogram_frame_impl(
        inp_data,
        FRAME_LEN,
        out_data->data,
        N_FFT_MEL,
        HOP_LENGTH_MEL,
        N_MELS
    );

    // 更新输出数据的维度信息
    out_data->row = N_MELS;
    out_data->col = NUM_MEL_FRAMES;
    out_data->channel = 1;

    return ALGO_NORMAL;
}

static int adaptive_avg_pool2d_to_1x1(const Conv2dData *input_feat, float *output_vector) {
    if (!input_feat || !input_feat->data || !output_vector) {
        return ALGO_POINTER_NULL;
    }
    if (input_feat->row == 0 || input_feat->col == 0) {
        for (uint16_t c = 0; c < input_feat->channel; ++c) {
            output_vector[c] = 0.0F;
        }

        return (input_feat->row == 0 && input_feat->col == 0 && input_feat->channel > 0) \
            ? ALGO_NORMAL : ALGO_DATA_EXCEPTION;
    }

    uint32_t num_elements_per_channel = (uint32_t)input_feat->row * input_feat->col;

    for (uint16_t c = 0; c < input_feat->channel; ++c) {
        float channel_sum = 0.0F;
        uint32_t channel_offset = c * num_elements_per_channel;
        for (uint32_t i = 0; i < num_elements_per_channel; ++i) {
            channel_sum += input_feat->data[channel_offset + i];
        }
        output_vector[c] = channel_sum / num_elements_per_channel;
    }
    return ALGO_NORMAL;
}


int vad_inference(Conv2dData *inp_data, float *speech_probability) {
    int ret = ALGO_NORMAL;
    *speech_probability = 0.0f; // 初始化


    Conv2dData current_out; // 用于存储各层输出
    Conv2dData prev_out = *inp_data; // 初始输入


//    float *buffer1 = vad_temp_conv_output_buf;
//    float *buffer2 = vad_temp_pool_output_buf;
    float *buffer1 = model_temp_conv_buf; // 修改后  cyw7
    float *buffer2 = model_temp_pool_buf; // 修改后

    BatchNorm2d vad_res1_bn1_params = {
    		.size = 8,
            .mean = (float*)model_0_bn1_running_mean,
            .var = (float*)model_0_bn1_running_var,
            .gamma = (float*)model_0_bn1_weight,
            .beta = (float*)model_0_bn1_bias
	};
    BatchNorm2d vad_res1_bn2_params = {
    		.size = 8,
            .mean = (float*)model_0_bn2_running_mean,
            .var = (float*)model_0_bn2_running_var,
            .gamma = (float*)model_0_bn2_weight,
            .beta = (float*)model_0_bn2_bias
    };
    BatchNorm2d vad_res1_bn_s_params = { .size = 8,
            .mean = (float*)model_0_shortcut_1_running_mean,
            .var = (float*)model_0_shortcut_1_running_var,
            .gamma = (float*)model_0_shortcut_1_weight,
            .beta = (float*)model_0_shortcut_1_bias
    };

    current_out.data = buffer1; // 输出到buffer1
    ret = residual_block_forward(
        &prev_out, &current_out,
		model_0_conv1_weight, &vad_res1_bn1_params,
		model_0_conv2_weight, &vad_res1_bn2_params,
        8,    // VAD ResBlock1 输出通道数
        true,
		model_0_shortcut_0_weight, &vad_res1_bn_s_params,
//        vad_res_main_path_buf, vad_res_main_path_buf,
//        vad_res_shortcut_path_buf, vad_res_padded_buf
		model_res_main_path_buf, model_res_main_path_buf, // 修改后  cyw7
		model_res_shortcut_path_buf, model_res_padded_buf // 修改后
    );
    if (ret != ALGO_NORMAL) goto func_exit_vad;
    prev_out = current_out; // 当前输出作为下一层输入

    // --- 第一个 MaxPool (输出8通道) ---
    current_out.data = buffer2; // 输出到buffer2
    current_out.channel = prev_out.channel;
    current_out.row = cal_conv_out_len(prev_out.row, 0, 2, 2);
    current_out.col = cal_conv_out_len(prev_out.col, 0, 2, 2);
    ret = max_pool2d(&prev_out, &current_out, 2, 2);
    if (ret != ALGO_NORMAL) goto func_exit_vad;
    prev_out = current_out;

    // --- 第二个 ResidualBlock ---
    BatchNorm2d vad_res2_bn1_params = {
    		.size = 24,
            .mean = (float*)model_2_bn1_running_mean,
            .var = (float*)model_2_bn1_running_var,
            .gamma = (float*)model_2_bn1_weight,
            .beta = (float*)model_2_bn1_bias
    };
    BatchNorm2d vad_res2_bn2_params = {
    		.size = 24,
            .mean = (float*)model_2_bn2_running_mean,
            .var = (float*)model_2_bn2_running_var,
            .gamma = (float*)model_2_bn2_weight,
            .beta = (float*)model_2_bn2_bias
    };
    BatchNorm2d vad_res2_bn_s_params = {
    		.size = 24,
            .mean = (float*)model_2_shortcut_1_running_mean,
            .var = (float*)model_2_shortcut_1_running_var,
            .gamma = (float*)model_2_shortcut_1_weight,
            .beta = (float*)model_2_shortcut_1_bias
    };

    current_out.data = buffer1; // 输出到buffer1 (复用)
    ret = residual_block_forward(
        &prev_out, &current_out,
		model_2_conv1_weight, &vad_res2_bn1_params,
		model_2_conv2_weight, &vad_res2_bn2_params,
        24,
        true,
		model_2_shortcut_0_weight, &vad_res2_bn_s_params,
//        vad_res_main_path_buf, vad_res_main_path_buf,
//        vad_res_shortcut_path_buf, vad_res_padded_buf
		model_res_main_path_buf, model_res_main_path_buf, // 修改后   cyw7
		model_res_shortcut_path_buf, model_res_padded_buf // 修改后
    );
    if (ret != ALGO_NORMAL) goto func_exit_vad;
    prev_out = current_out;

    // --- 第二个 MaxPool ---
    current_out.data = buffer2; // 输出到buffer2 (复用)
    current_out.channel = prev_out.channel;
    current_out.row = cal_conv_out_len(prev_out.row, 0, 2, 2);
    current_out.col = cal_conv_out_len(prev_out.col, 0, 2, 2);
    ret = max_pool2d(&prev_out, &current_out, 2, 2);
    if (ret != ALGO_NORMAL) goto func_exit_vad;
    prev_out = current_out;

    // --- AdaptiveAvgPool2d ---
//    ret = adaptive_avg_pool2d_to_1x1(&prev_out, vad_temp_adaptive_pool_out_buf);
    ret = adaptive_avg_pool2d_to_1x1(&prev_out, model_temp_adaptive_pool_out_buf); // 修改后  cyw7
    if (ret != ALGO_NORMAL) goto func_exit_vad;

    // --- Linear Layer (输出2类: 无语音, 有语音) ---
    float vad_linear_out[2];
    LinearParam linear_config_vad = {
        .inp_size = 24, // VAD模型最后一个卷积/池化层的输出通道数
        .fea_size = 2,  // 二分类
        .weight   = (float*)output_1_weight, // 对应VAD模型的output.1.weight
        .bias     = (float*)output_1_bias
    };
    ret = linear_layer(model_temp_adaptive_pool_out_buf, &linear_config_vad, vad_linear_out); // 修改后  cyw7
    if (ret != ALGO_NORMAL) goto func_exit_vad;

    // --- Softmax ---
    float exp_logit0 = expf(vad_linear_out[0]); // 无语音
    float exp_logit1 = expf(vad_linear_out[1]); // 有语音
    float sum_exp = exp_logit0 + exp_logit1;

    if (sum_exp > MEL_EPS) {
        *speech_probability = exp_logit1 / sum_exp;
    } else {
        *speech_probability = 0.5f;
    }

func_exit_vad:
    return ret;
}

#define SC_INPUT_C 1
#define SC_INPUT_H N_MELS
#define SC_INPUT_W 19  //cyw8
#define SC_INPUT_SIZE (SC_INPUT_C * SC_INPUT_H * SC_INPUT_W) // 1 * 32 * 5 = 160

// ResBlock1 (model_seq.0) 输出
#define SC_RES1_OUT_C 8
#define SC_RES1_OUT_H SC_INPUT_H
#define SC_RES1_OUT_W SC_INPUT_W
#define SC_RES1_OUT_SIZE (SC_RES1_OUT_C * SC_RES1_OUT_H * SC_RES1_OUT_W) // 8 * 32 * 5 = 1280

// Pool1 (model_seq.1) 输出
#define SC_POOL1_OUT_C SC_RES1_OUT_C
#define SC_POOL1_OUT_H (SC_RES1_OUT_H / 2)
#define SC_POOL1_OUT_W (SC_RES1_OUT_W / 2)
#define SC_POOL1_OUT_SIZE (SC_POOL1_OUT_C * SC_POOL1_OUT_H * SC_POOL1_OUT_W) // 8 * 16 * 2 = 256

// ResBlock2 (model_seq.2) 输出
#define SC_RES2_OUT_C 24
#define SC_RES2_OUT_H SC_POOL1_OUT_H
#define SC_RES2_OUT_W SC_POOL1_OUT_W
#define SC_RES2_OUT_SIZE (SC_RES2_OUT_C * SC_RES2_OUT_H * SC_RES2_OUT_W) // 24 * 16 * 2 = 768

// Pool2 (model_seq.3) 输出
#define SC_POOL2_OUT_C SC_RES2_OUT_C
#define SC_POOL2_OUT_H (SC_RES2_OUT_H / 2)
#define SC_POOL2_OUT_W (SC_RES2_OUT_W / 2)
#define SC_POOL2_OUT_SIZE (SC_POOL2_OUT_C * SC_POOL2_OUT_H * SC_POOL2_OUT_W) // 24 * 8 * 1 = 192

// --- 计算最大缓冲区大小 ---
// 用于 speaker_classifier_inference 内部，存储卷积/残差块的输出 (复用)
#define SC_TEMP_CONV_OUTPUT_BUF_SIZE ( (SC_RES1_OUT_SIZE > SC_RES2_OUT_SIZE) ? SC_RES1_OUT_SIZE : SC_RES2_OUT_SIZE )

// 用于 speaker_classifier_inference 内部，存储池化层的输出 (复用)
#define SC_TEMP_POOL_OUTPUT_BUF_SIZE ( (SC_POOL1_OUT_SIZE > SC_POOL2_OUT_SIZE) ? SC_POOL1_OUT_SIZE : SC_POOL2_OUT_SIZE )

// 用于 speaker_classifier_inference 内部，卷积前的padding (复用)
#define SC_PAD_AMOUNT 1
#define SC_PADDED_RES1_CONV1_SIZE (SC_INPUT_C * (SC_INPUT_H + 2*SC_PAD_AMOUNT) * (SC_INPUT_W + 2*SC_PAD_AMOUNT))
#define SC_PADDED_RES1_CONV2_SIZE (SC_RES1_OUT_C * (SC_RES1_OUT_H + 2*SC_PAD_AMOUNT) * (SC_RES1_OUT_W + 2*SC_PAD_AMOUNT))
#define SC_PADDED_RES2_CONV1_SIZE (SC_POOL1_OUT_C * (SC_POOL1_OUT_H + 2*SC_PAD_AMOUNT) * (SC_POOL1_OUT_W + 2*SC_PAD_AMOUNT))
#define SC_PADDED_RES2_CONV2_SIZE (SC_RES2_OUT_C * (SC_RES2_OUT_H + 2*SC_PAD_AMOUNT) * (SC_RES2_OUT_W + 2*SC_PAD_AMOUNT))

#define SC_TEMP_MAX_PADDED_1 ( (SC_PADDED_RES1_CONV1_SIZE > SC_PADDED_RES1_CONV2_SIZE) ? SC_PADDED_RES1_CONV1_SIZE : SC_PADDED_RES1_CONV2_SIZE )
#define SC_TEMP_MAX_PADDED_2 ( (SC_PADDED_RES2_CONV1_SIZE > SC_PADDED_RES2_CONV2_SIZE) ? SC_PADDED_RES2_CONV1_SIZE : SC_PADDED_RES2_CONV2_SIZE )
#define SC_RES_PADDED_BUF_SIZE ( (SC_TEMP_MAX_PADDED_1 > SC_TEMP_MAX_PADDED_2) ? SC_TEMP_MAX_PADDED_1 : SC_TEMP_MAX_PADDED_2 ) // Max = 15840

// 用于 residual_block_forward 内部的主路径和快捷连接路径的临时存储
#define SC_RES_INTERNAL_TEMP_BUF_SIZE SC_TEMP_CONV_OUTPUT_BUF_SIZE

// Adaptive pool output buffer (大小是最后一个卷积层的输出通道数,即ResBlock2的输出通道数)
#define SC_ADAPTIVE_POOL_OUT_VEC_SIZE SC_RES2_OUT_C // 24


int residual_block_forward(
    Conv2dData *input_feat,
    Conv2dData *output_feat,

    // 主路径参数
    const float *conv1_w, BatchNorm2d *bn1_params,
    const float *conv2_w, BatchNorm2d *bn2_params,
    uint16_t main_out_channels,

    // 快捷连接路径参数
    bool has_shortcut_conv,
    const float *conv_s_w, BatchNorm2d *bn_s_params,

    // 临时缓冲区 (从外部传入或使用全局静态)
    float *temp_main_path_buf1,
    float *temp_main_path_buf2,
    float *temp_shortcut_path_buf,
    float *temp_padded_buf
) {
    int ret = ALGO_NORMAL;
    Conv2dData current_main_out;
    Conv2dData current_shortcut_out;

    uint16_t pad_amount_conv = 1;

    Conv2dData padded_input_conv1;
    padded_input_conv1.row = input_feat->row + 2 * pad_amount_conv;
    padded_input_conv1.col = input_feat->col + 2 * pad_amount_conv;
    padded_input_conv1.channel = input_feat->channel;
    padded_input_conv1.data = temp_padded_buf;
    padding_value(input_feat, pad_amount_conv, 0.0F, padded_input_conv1.data);

    Conv2dFilter filter_conv1 = {
        .channel = input_feat->channel, .col = 3, .row = 3,
        .filter_num = main_out_channels, .data = conv1_w
    };
    Conv2dConfig config_conv1 = { .stride = 1, .pad = 0, .filter = &filter_conv1, .bn = bn1_params };

    current_main_out.channel = main_out_channels;
    current_main_out.row = cal_conv_out_len(padded_input_conv1.row, 0, 3, 1);
    current_main_out.col = cal_conv_out_len(padded_input_conv1.col, 0, 3, 1);
    current_main_out.data = temp_main_path_buf1; // 使用临时缓冲区

    ret = conv2d_bn_no_bias(&padded_input_conv1, &config_conv1, &current_main_out);
    if (ret != ALGO_NORMAL) return ret;
    ret = leaky_relu(0.01F, current_main_out.data,
                     current_main_out.channel * current_main_out.row * current_main_out.col,
                     current_main_out.data);
    if (ret != ALGO_NORMAL) return ret;

    Conv2dData padded_input_conv2;
    padded_input_conv2.row = current_main_out.row + 2 * pad_amount_conv;
    padded_input_conv2.col = current_main_out.col + 2 * pad_amount_conv;
    padded_input_conv2.channel = current_main_out.channel;
    padded_input_conv2.data = temp_padded_buf;
    padding_value(&current_main_out, pad_amount_conv, 0.0F, padded_input_conv2.data);

    Conv2dFilter filter_conv2 = {
        .channel = main_out_channels, .col = 3, .row = 3,
        .filter_num = main_out_channels, .data = conv2_w
    };
    Conv2dConfig config_conv2 = { .stride = 1, .pad = 0, .filter = &filter_conv2, .bn = bn2_params };

    // 主路径最终输出，存到 temp_main_path_buf2
    Conv2dData main_path_final_out;
    main_path_final_out.channel = main_out_channels;
    main_path_final_out.row = cal_conv_out_len(padded_input_conv2.row, 0, 3, 1);
    main_path_final_out.col = cal_conv_out_len(padded_input_conv2.col, 0, 3, 1);
    main_path_final_out.data = temp_main_path_buf2;

    ret = conv2d_bn_no_bias(&padded_input_conv2, &config_conv2, &main_path_final_out);
    if (ret != ALGO_NORMAL) return ret;

    // --- 2. 快捷连接路径 ---
    if (has_shortcut_conv) { // in_channels != out_channels
        // 1x1 Conv + BN
        Conv2dFilter filter_s = {
            .channel = input_feat->channel, .col = 1, .row = 1,
            .filter_num = main_out_channels, .data = conv_s_w
        };
        // BN_shortcut 的参数是 bn_s_params
        Conv2dConfig config_s = { .stride = 1, .pad = 0, .filter = &filter_s, .bn = bn_s_params };

        current_shortcut_out.channel = main_out_channels;
        current_shortcut_out.row = cal_conv_out_len(input_feat->row, 0, 1, 1);
        current_shortcut_out.col = cal_conv_out_len(input_feat->col, 0, 1, 1);
        current_shortcut_out.data = temp_shortcut_path_buf; // 使用快捷连接的临时缓冲区

        ret = conv2d_bn_no_bias(input_feat, &config_s, &current_shortcut_out);
        if (ret != ALGO_NORMAL) return ret;
    } else {
        memcpy(temp_shortcut_path_buf, input_feat->data,
               input_feat->channel * input_feat->row * input_feat->col * sizeof(float));
        current_shortcut_out.channel = input_feat->channel;
        current_shortcut_out.row     = input_feat->row;
        current_shortcut_out.col     = input_feat->col;
        current_shortcut_out.data    = temp_shortcut_path_buf;
    }

    if (main_path_final_out.channel != current_shortcut_out.channel ||
        main_path_final_out.row != current_shortcut_out.row ||
        main_path_final_out.col != current_shortcut_out.col) {
        return ALGO_DATA_EXCEPTION;
    }
    int add_size = main_path_final_out.channel * main_path_final_out.row * main_path_final_out.col;
    //element_wise_add(main_path_final_out.data, current_shortcut_out.data, output_feat->data, add_size);
    riscv_add_f32 (main_path_final_out.data, current_shortcut_out.data, output_feat->data, add_size);

    output_feat->channel = main_path_final_out.channel;
    output_feat->row = main_path_final_out.row;
    output_feat->col = main_path_final_out.col;

    // --- 4. 最终 LeakyReLU
    ret = leaky_relu(0.01F, output_feat->data, add_size, output_feat->data);
    if (ret != ALGO_NORMAL) return ret;

    return ALGO_NORMAL;
}

int speaker_classifier_inference(Conv2dData *inp_data, float probabilities[5]) {
    int ret = ALGO_NORMAL;
    for(int i=0; i<5; ++i) probabilities[i] = 0.0f;

    Conv2dData res1_out, pool1_out;
    Conv2dData res2_out, pool2_out;
    Conv2dData adaptive_pool_in;

//    res1_out.data = sc_temp_conv_output_buf;
//    pool1_out.data = sc_temp_pool_output_buf;
//    res2_out.data = sc_temp_conv_output_buf;
//    pool2_out.data = sc_temp_pool_output_buf;
    res1_out.data = model_temp_conv_buf; // 修改后  cyw7
    pool1_out.data = model_temp_pool_buf; // 修改后
    res2_out.data = model_temp_conv_buf; // 修改后
    pool2_out.data = model_temp_pool_buf; // 修改后

    BatchNorm2d res1_bn1_params = {
        .size = 8,
        .mean = (float*)speaker_model_model_0_bn1_running_mean,
        .var = (float*)speaker_model_model_0_bn1_running_var,
        .gamma = (float*)speaker_model_model_0_bn1_weight,
        .beta = (float*)speaker_model_model_0_bn1_bias
    };
    BatchNorm2d res1_bn2_params = {
        .size = 8,
        .mean = (float*)speaker_model_model_0_bn2_running_mean,
        .var = (float*)speaker_model_model_0_bn2_running_var,
        .gamma = (float*)speaker_model_model_0_bn2_weight,
        .beta = (float*)speaker_model_model_0_bn2_bias
    };
    BatchNorm2d res1_bn_s_params = { // Shortcut BN for ResBlock1
        .size = 8,
        .mean = (float*)speaker_model_model_0_shortcut_1_running_mean,
        .var = (float*)speaker_model_model_0_shortcut_1_running_var,
        .gamma = (float*)speaker_model_model_0_shortcut_1_weight,
        .beta = (float*)speaker_model_model_0_shortcut_1_bias
    };

    ret = residual_block_forward(
        inp_data, &res1_out,
        speaker_model_model_0_conv1_weight, &res1_bn1_params,
        speaker_model_model_0_conv2_weight, &res1_bn2_params,
        8,
        true,
		speaker_model_model_0_shortcut_0_weight, &res1_bn_s_params,
//        sc_res_main_path_buf, sc_res_main_path_buf,
//        sc_res_shortcut_path_buf, sc_res_padded_buf
		model_res_main_path_buf, model_res_main_path_buf, // 修改后  cyw7
		model_res_shortcut_path_buf, model_res_padded_buf // 修改后
    );
    if (ret != ALGO_NORMAL) goto func_exit_speaker;

    // === MaxPool2d (model_seq.1) ===
    pool1_out.channel = res1_out.channel;
    pool1_out.row = cal_conv_out_len(res1_out.row, 0, 2, 2);
    pool1_out.col = cal_conv_out_len(res1_out.col, 0, 2, 2);
    ret = max_pool2d(&res1_out, &pool1_out, 2, 2);
    if (ret != ALGO_NORMAL) goto func_exit_speaker;

    // === 第二层: ResidualBlock(in_channels=8, out_channels=24) ===
    BatchNorm2d res2_bn1_params = {
        .size = 24,
        .mean = (float*)speaker_model_model_2_bn1_running_mean,
        .var = (float*)speaker_model_model_2_bn1_running_var,
        .gamma = (float*)speaker_model_model_2_bn1_weight,
        .beta = (float*)speaker_model_model_2_bn1_bias
    };
    BatchNorm2d res2_bn2_params = {
        .size = 24,
        .mean = (float*)speaker_model_model_2_bn2_running_mean,
        .var = (float*)speaker_model_model_2_bn2_running_var,
        .gamma = (float*)speaker_model_model_2_bn2_weight,
        .beta = (float*)speaker_model_model_2_bn2_bias
    };
    BatchNorm2d res2_bn_s_params = { // Shortcut BN for ResBlock2
        .size = 24,
        .mean = (float*)speaker_model_model_2_shortcut_1_running_mean,
        .var = (float*)speaker_model_model_2_shortcut_1_running_var,
        .gamma = (float*)speaker_model_model_2_shortcut_1_weight,
        .beta = (float*)speaker_model_model_2_shortcut_1_bias
    };

    ret = residual_block_forward(
        &pool1_out, &res2_out,
		speaker_model_model_2_conv1_weight, &res2_bn1_params,
		speaker_model_model_2_conv2_weight, &res2_bn2_params,
        24,
        true,
		speaker_model_model_2_shortcut_0_weight, &res2_bn_s_params,
//        sc_res_main_path_buf, sc_res_main_path_buf,
//        sc_res_shortcut_path_buf, sc_res_padded_buf
		model_res_main_path_buf, model_res_main_path_buf, // 修改后    cyw7
		model_res_shortcut_path_buf, model_res_padded_buf // 修改后
    );
    if (ret != ALGO_NORMAL) goto func_exit_speaker;

    // === MaxPool2d (model_seq.3) ===
    pool2_out.channel = res2_out.channel;
    pool2_out.row = cal_conv_out_len(res2_out.row, 0, 2, 2);
    pool2_out.col = cal_conv_out_len(res2_out.col, 0, 2, 2);
    ret = max_pool2d(&res2_out, &pool2_out, 2, 2);
    if (ret != ALGO_NORMAL) goto func_exit_speaker;

    // === AdaptiveAvgPool2d ===
    adaptive_pool_in = pool2_out;
//    ret = adaptive_avg_pool2d_to_1x1(&adaptive_pool_in, sc_temp_adaptive_pool_out_buf);
    ret = adaptive_avg_pool2d_to_1x1(&adaptive_pool_in, model_temp_adaptive_pool_out_buf); // 修改后  cyw7
    if (ret != ALGO_NORMAL) goto func_exit_speaker;

    // === Linear Layer (output_fc.1) ===
    float final_linear_out[5];   //cyw8,5分类
    LinearParam linear_config_speaker = {
        .inp_size = 24,
        .fea_size = 5,   //cyw8
        .weight   = (float*)speaker_model_output_1_weight,
        .bias     = (float*)speaker_model_output_1_bias
    };
//    ret = linear_layer(sc_temp_adaptive_pool_out_buf, &linear_config_speaker, final_linear_out);
    ret = linear_layer(model_temp_adaptive_pool_out_buf, &linear_config_speaker, final_linear_out); // 修改后
    if (ret != ALGO_NORMAL) goto func_exit_speaker;

    // === Softmax ===
//    float exp_class0 = expf(final_linear_out[0]);
//    float exp_class1 = expf(final_linear_out[1]);
//    float exp_class2 = expf(final_linear_out[2]);
//    float sum_exp = exp_class0 + exp_class1 + exp_class2;
//
//    if (sum_exp > MEL_EPS) {
//        probabilities[0] = exp_class0 / sum_exp;
//        probabilities[1] = exp_class1 / sum_exp;
//        probabilities[2] = exp_class2 / sum_exp;
//    } else {
//        probabilities[0] = 1.0f / 3.0f;
//        probabilities[1] = 1.0f / 3.0f;
//        probabilities[2] = 1.0f / 3.0f;
//    }
    float exp_logits[5];   //cyw8
    float sum_exp = 0.0f;
    for (int i = 0; i < 5; i++) {
        exp_logits[i] = expf(final_linear_out[i]);
        sum_exp += exp_logits[i];
    }

    if (sum_exp > MEL_EPS) {
        for (int i = 0; i < 5; i++) {
            probabilities[i] = exp_logits[i] / sum_exp;
        }
    } else {
        for (int i = 0; i < 5; i++) {
            probabilities[i] = 1.0f / 5.0f;
        }
    }

func_exit_speaker:
    return ret;
}


