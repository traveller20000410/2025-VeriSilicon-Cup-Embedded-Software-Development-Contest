/*
 * constants.h
 *
 *  Created on: 2025年7月21日
 *      Author: 廖先强
 */

#ifndef GALAXY_SDK_CONSTANTS_H_
#define GALAXY_SDK_CONSTANTS_H_


#include "vpi_event.h"
#include "hal_pdm.h"

// 定义事件ID
#define EVT_PCM_DATA_READY 				(EVENT_SDK_END + 1)				// PCM数据准备就绪事件
#define EVT_ALGO_RESULT_READY 			(EVENT_SDK_END + 2)				// 算法结果准备就绪事件 (algo_task通知)
#define EVT_PRINT_DATA_READY			(EVENT_SDK_END + 3)
#define EVT_RAW_PCM_FOR_BLE             (EVENT_SDK_END + 4)             // 原始PCM数据用于蓝牙发送的事件
#define EVENT_ALGO_RESULT_READY         (EVENT_SDK_END + 5)             /**< 通知BLE任务：有新的算法结果需要发送 */

// 通过事件传递PCM数据的结构体
typedef struct {
    uint8_t* pcm_data;
    uint32_t data_size_bytes;
} PcmDataInfo;

// 定义环形缓冲区结构体
typedef struct
{
    uint8_t *buffer;    // 缓冲区起始地址
    uint32_t size;      // 缓冲区总大小
    uint32_t head;      // 写入指针 (下一个写入的位置)
    uint32_t tail;      // 读取指针 (下一个读取的位置)
    uint32_t count;     // 当前缓冲区中数据的字节数
} CircularBuffer;

// 环形缓冲区操作函数原型
void circularbuffer_init(CircularBuffer *cb, uint8_t *buf, uint32_t size);
uint32_t circularbuffer_write(CircularBuffer *cb, const uint8_t *data, uint32_t len);
uint32_t circularbuffer_read(CircularBuffer *cb, uint8_t *data, uint32_t len);
uint32_t circularbuffer_getusedsize(CircularBuffer *cb);

#define PDM_BUFFER_SIZE_BYTES (8000)


// Task Stack Sizes
#define SPEECH_TASK_STACK_SIZE          (1024 * 0.5) // 2KB
#define ALGO_TASK_STACK_SIZE            (1024 * 2) // 4KB
#define BLE_TASK_STACK_SIZE 			(4 * 1024)

#pragma region algo
#define FS                      (8000)
#define FRAME_STEP              (4000) //cyw8
#define FRAME_LEN               (4000) //cyw8
#define BN_EPS                  (1e-5f)

// --- 梅尔频谱图参数 ---
#define win_length              512
#define N_FFT_MEL               512
#define HOP_LENGTH_MEL          185
#define N_MELS                  32
#define NUM_MEL_FRAMES          (int)((FRAME_LEN - win_length) / HOP_LENGTH_MEL + 1) //19
#define MEL_EPS (1e-9f)

#pragma endregion

typedef enum AlgoError {
    ALGO_ERR_GENERIC = -1,
    ALGO_NORMAL      = 0,
    ALGO_MALLOC_FAIL,
    ALGO_DATA_NOT_ENOUGH,
    ALGO_DATA_EXCEPTION,
    ALGO_DATA_NULL,
    ALGO_DATA_INVALID,
    ALGO_DATA_TOO_MANY,
    ALGO_ERR_LOCAL_FREE,
    ALGO_POINTER_NULL,
    ALGO_DATA_QUALITY_POOR,
    ALGO_IO_EXCEPTION
} AlgoError;

// 定义事件参数
typedef enum {
    APP_CMD_ACTION_START,
    APP_CMD_ACTION_STOP,
} AppCmdAction;

typedef struct {
    EventManagerId target_mgr;
    AppCmdAction action;
} AppCmd;




#endif /* GALAXY_SDK_CONSTANTS_H_ */
