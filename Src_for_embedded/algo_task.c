/*
 * algo_task.c (Refactored for 0.5s direct inference)
 *
 *  Created on: 2025年7月21日
 *      Author: 廖先强
 */

#include "algo_task.h"
#include "FreeRTOS.h"
#include "task.h"
#include "vpi_event.h"
#include <string.h>
#include <stdio.h>
#include "vpi_error.h"
#include "speech_task.h"
#include "conv.h"
#include "vad.h"

// --- 全局/静态变量 ---
TaskHandle_t algo_task_handle;
static EventManager algo_event_manager;
//static EventManager print_event_manager;

// 音频流处理相关的静态缓冲区
static float algo_total_data_buf[FRAME_LEN]; // 用于存放归一化后的0.5秒音频 (4000个float)
static float algo_vad_inp_data_buf[N_MELS * NUM_MEL_FRAMES]; // 梅尔频谱图数据 (32 * 19)
static uint8_t pcm_frame_buffer_static[PDM_BUFFER_SIZE_BYTES]; // 从环形缓冲区读取的原始PCM数据

// 用于在任务间传递结果的结构体
typedef struct {
    uint64_t timestamp_samples; // 帧结束时的时间戳（以采样点计）
    int predicted_class_id;     // 最终预测的类别ID
} InferenceResult;

// 用于打印任务的静态结果存储
//static InferenceResult result_to_print;

// 类别映射表 (6个类别: 无人声 + 5个说话人)
const char* CLASS_MAPPING_FOR_BLE[] = {
    "No Human Voice",
    "Speaker ID:XiaoXin",
    "Speaker ID:XiaoYuan",
    "Speaker ID:XiaoSi",
    "Speaker ID:XiaoLai",
    "Speaker ID:Others"
};

// 全局变量，跟踪当前处理的音频在整个流中的偏移量
static uint64_t current_audio_stream_offset_samples = 0;

// --- 函数实现 ---

/**
 * @brief 将从PDM驱动读取的16位PCM数据分割并归一化为浮点数。
 */
//static void datasplit_norm(uint8_t *pcm_data, uint32_t data_size)
//{
//    uint32_t num_samples = data_size / 2;
//    float max_val = 0.0f;
//
//    // 分割并寻找最大绝对值
//    for (uint32_t i = 0; i < num_samples; i++) {
//        int16_t sample = (pcm_data[i * 2 + 1] << 8) | pcm_data[i * 2];
//        algo_total_data_buf[i] = (float)sample;
//        if (fabsf(algo_total_data_buf[i]) > max_val) {
//            max_val = fabsf(algo_total_data_buf[i]);
//        }
//    }
//
//    // 归一化
//    if (max_val > 0) {
//        for (uint32_t i = 0; i < num_samples; i++) {
//            algo_total_data_buf[i] /= max_val;
//        }
//    }
//}

static void datasplit_to_float(uint8_t *pcm_data, uint32_t data_size)
{
    // data_size 是字节数，一个采样点是2字节(int16_t)
    uint32_t num_samples = data_size / 2;

    // 直接进行类型转换
    for (uint32_t i = 0; i < num_samples; i++) {
        // 从小端字节序数据重组为16位有符号整数
        int16_t sample = (int16_t)((pcm_data[i * 2 + 1] << 8) | pcm_data[i * 2]);
        // 将其转换为float并存入全局缓冲区
        algo_total_data_buf[i] = (float)sample;
    }
}

/**
 * @brief 确保所有算法资源只被初始化一次。
 */
static void algo_task_init_once()
{
    static int initialized = 0;
    if (!initialized) {
        if (initialize_mel_resources() != ALGO_NORMAL) {
            printf("CRITICAL: Failed to initialize Mel resources!\n");
        }

        fft_module_init(N_FFT_MEL); // from vad.h, 初始化FFT模块  cyw8

        current_audio_stream_offset_samples = 0;
        initialized = 1;
        initialize_hann_window_coeffs(); // 初始化汉宁窗
        printf("Mel resources and algo state initialized.\n");
    }
}

/**
 * @brief 处理一个0.5秒的音频块，进行VAD和声纹识别，并准备好结果。
 */
static ResultDataParam result_param;
static void process_audio_chunk(float *pcm_chunk_data)
{
    int ret;
    Conv2dData mel_spec_data;
    mel_spec_data.channel = 1;
    mel_spec_data.row     = N_MELS;
    mel_spec_data.col     = NUM_MEL_FRAMES;
    mel_spec_data.data    = algo_vad_inp_data_buf;

    // 1. 预处理：计算梅尔频谱图
    ret = vad_preprocess(pcm_chunk_data, &mel_spec_data);
    if (ret != ALGO_NORMAL) {
        printf("Error: vad_preprocess failed (%d)\n", ret);
        return;
    }

    // 2. VAD 推理
    float vad_prob = 0.0f;
    ret = vad_inference(&mel_spec_data, &vad_prob);
    if (ret != ALGO_NORMAL) {
        printf("Error: vad_inference failed (%d)\n", ret);
        return;
    }

    int final_class_id = 0; // 默认为 "No Human Voice"

    // 3. 如果VAD检测到语音，则进行声纹分类
    if (vad_prob > 0.5) {
        float speaker_probs[5] = {0.0f}; // 5个分类
        ret = speaker_classifier_inference(&mel_spec_data, speaker_probs);
        if (ret != ALGO_NORMAL) {
            printf("Warning: speaker_classifier_inference failed (%d). Classifying as 'Others'.\n", ret);
            final_class_id = 5; // 如果分类失败，保守地标记为"Others" (类别ID 5)
        } else {
            // 找出概率最高的类别
            int8_t max_prob_idx = 0;
            for (int8_t i = 1; i < 5; i++) {
                if (speaker_probs[i] > speaker_probs[max_prob_idx]) {
                    max_prob_idx = i;
                }
            }

//            // 类别ID映射：模型输出 0-4 对应于类别 1-5
//            final_class_id = max_prob_idx + 1;
            float max_prob = speaker_probs[max_prob_idx];

                        // 如果最高概率不足阈值，就归为 Others
                const float SPEAKER_CONF_THRESHOLD = 0.5f;  // 根据需求可调
                if (max_prob < SPEAKER_CONF_THRESHOLD) {
                    final_class_id = 5;  // Others
                    } else {
                        // 否则才按概率最大者归类：模型输出 0-4 对应类别 1-5
                        final_class_id = max_prob_idx + 1;
                    }
        }
    }

//    // 4. 准备结果并发送事件
//    result_to_print.timestamp_samples = current_audio_stream_offset_samples + FRAME_LEN;
//    result_to_print.predicted_class_id = final_class_id;
//    vpi_event_notify(EVT_PRINT_DATA_READY, &result_to_print);
    if (final_class_id >= 0 && final_class_id < 6) {
        const char* result_str = CLASS_MAPPING_FOR_BLE[final_class_id];

//        ResultDataParam result_param;
        result_param.data = (uint8_t*)result_str;
        result_param.length = strlen(result_str);

        // 发送算法结果事件
        vpi_event_notify(EVENT_ALGO_RESULT_READY, &result_param);}
}


/**
 * @brief FreeRTOS事件回调函数，处理PCM数据准备就绪的事件。
 */
// 定义算法运行状态
typedef enum {
    ALGO_STATE_STOPPED,
    ALGO_STATE_RUNNING
} AlgoState;

static volatile AlgoState g_algo_state = ALGO_STATE_STOPPED;
uint8_t g_ble_raw_pcm_buffer[8000];
static int pcm_data_event_handler(EventManager manager, EventId event_id, EventParam param)
{
    if (event_id == EVENT_CMD_UPDATE) {
        AppCmd *cmd = (AppCmd *)param;
        // 确认是发给自己的事件
        if (cmd && cmd->target_mgr == EVENT_MGR_ALGO) {
            if (cmd->action == APP_CMD_ACTION_START) {
                if (g_algo_state == ALGO_STATE_STOPPED) {
                    g_algo_state = ALGO_STATE_RUNNING;
                }
            } else if (cmd->action == APP_CMD_ACTION_STOP) {
                if (g_algo_state == ALGO_STATE_RUNNING) {
                    g_algo_state = ALGO_STATE_STOPPED;
                }
            }
        }
    }
    if (event_id == EVT_PCM_DATA_READY) {
        //algo_task_init_once();

        // 只要环形缓冲区中有足够一个完整帧的数据，就持续处理
        while (circularbuffer_getusedsize(&speech_pcm_circular_buffer) >= PDM_BUFFER_SIZE_BYTES) {

            uint32_t bytes_read = circularbuffer_read(
                &speech_pcm_circular_buffer,
                pcm_frame_buffer_static,
                PDM_BUFFER_SIZE_BYTES
            );

            if (bytes_read == PDM_BUFFER_SIZE_BYTES) {
				memcpy(g_ble_raw_pcm_buffer, pcm_frame_buffer_static, bytes_read);
                vpi_event_notify(EVT_RAW_PCM_FOR_BLE, NULL);
                // 更新时间戳
                current_audio_stream_offset_samples += FRAME_STEP;
                if(g_algo_state == ALGO_STATE_STOPPED)
                {
                	return 0;
                }
                // 归一化PCM数据到 algo_total_data_buf
//                datasplit_norm(pcm_frame_buffer_static, bytes_read);
                datasplit_to_float(pcm_frame_buffer_static, bytes_read);//cyw8

                // 直接处理整个0.5秒的数据块
                process_audio_chunk(algo_total_data_buf);

            } else {
                // 这种情况理论上不应发生，除非有严重的数据同步问题
                printf("Error: Could not read a full PCM frame! Bytes read: %lu\n", bytes_read);
                break;
            }
        }
    }
    return EVENT_OK;
}

/**
 * @brief FreeRTOS事件回调函数，处理打印请求事件。
 */
//static int print_event_handler(EventManager manager, EventId event_id, EventParam param)
//{
//    if (event_id == EVT_PRINT_DATA_READY && param != NULL) {
//        InferenceResult* result = (InferenceResult*)param;
//        uint32_t end_time_ms = (uint32_t)((double)result->timestamp_samples * 1000.0 / FS);
//
//        // 确保类别ID在有效范围内
//        if (result->predicted_class_id >= 0 && result->predicted_class_id < sizeof(CLASS_MAPPING_C)/sizeof(CLASS_MAPPING_C[0])) {
//             printf("%lu ms, %s\n", end_time_ms, CLASS_MAPPING_C[result->predicted_class_id]);
//        } else {
//             printf("%lu ms, Unknown Class ID: %d\n", end_time_ms, result->predicted_class_id);
//        }
//    }
//    return EVENT_OK;
//}

/**
 * @brief algo_task的任务入口函数。
 */
void algo_task_entry(void *param)
{
    algo_task_handle = xTaskGetCurrentTaskHandle();
    printf("Algo Task Initialized.\n");

    algo_event_manager = vpi_event_new_manager(EVENT_MGR_ALGO, pcm_data_event_handler);
    if (!algo_event_manager) {
        printf("Failed to create algo event manager!\n");
        return;
    }

    if (vpi_event_register(EVT_PCM_DATA_READY, algo_event_manager) != EVENT_OK) {
        printf("Failed to register EVT_PCM_DATA_READY!\n");
        return;
    }
    if (vpi_event_register(EVENT_CMD_UPDATE, algo_event_manager) != EVENT_OK) {
        printf("Failed to register EVENT_CMD_UPDATE for algo_task!\n");
        return;
    }
    algo_task_init_once();
    printf("Algo Task listening for events.\n");
    while(1) {
        if (vpi_event_listen(algo_event_manager) != EVENT_OK) {
            printf("Algo Task event listen failed!\n");
            // 考虑是否需要销毁任务或重启
        }
    }
}

/**
 * @brief print_task的任务入口函数。
 */
//void print_task_entry(void *param)
//{
//    print_event_manager = vpi_event_new_manager(EVENT_MGR_CUSTOM, print_event_handler);
//    if (!print_event_manager) {
//        printf("Failed to create print event manager!\n");
//        return;
//    }
//
//    if (vpi_event_register(EVT_PRINT_DATA_READY, print_event_manager) != EVENT_OK) {
//        printf("Failed to register EVT_PRINT_DATA_READY for print_task!\n");
//        return;
//    }
//
//    printf("Print Task listening for events.\n");
//    while(1) {
//        if (vpi_event_listen(print_event_manager) != EVENT_OK) {
//            printf("Print Task event listen failed!\n");
//        }
//    }
//}
