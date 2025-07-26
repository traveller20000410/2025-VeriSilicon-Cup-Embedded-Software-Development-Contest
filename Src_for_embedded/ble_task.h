/*
 * ble_task.h
 *
 *  Created on: 2025年7月23日
 *      Author: 廖先强
 */

#ifndef GALAXY_SDK_BLE_TASK_H_
#define GALAXY_SDK_BLE_TASK_H_

#include <stdbool.h>
#include <stdint.h>
#include "vpi_event.h" // 包含事件系统头文件
#include "ble_connection.h"

// --- 定义包头参数 ---

// Block Type 定义
#define BLOCK_TYPE_RAW_DATA 0x01

// Sensor Type 定义
typedef enum
{
    SENSOR_TYPE_NONE = 0,
    SENSOR_TYPE_PPG = 1,
    SENSOR_TYPE_ECG = 2,
    SENSOR_TYPE_IMU = 3,
    SENSOR_TYPE_TMP = 4,
    SENSOR_TYPE_MIC = 5,
} SensorType;

// Data Subtype 定义 (这里只定义我们用到的)
#define DATA_SUBTYPE_MIC_RAW 0x01

// Data Format - Field 1 (低4位) 定义
typedef enum
{
    DATA_FORMAT_TYPE_UINT32 = 0,
    DATA_FORMAT_TYPE_INT32 = 1,
    DATA_FORMAT_TYPE_UINT16 = 2,
    DATA_FORMAT_TYPE_INT16 = 3,
    DATA_FORMAT_TYPE_UINT24 = 4,
    DATA_FORMAT_TYPE_INT24 = 5,
} DataFormatType;

// Data Format - Field 2 (高4位) 定义
// 这个值代表一个样本里包含多少个测量单元，对于单声道音频，它是1
#define DATA_FORMAT_UNITS_PER_SAMPLE_1 1

// Flag 比特位定义
#define FLAG_BIT_START_OF_COLLECTION (1 << 2) // bit 2: 标志数据采集的开始
#define FLAG_BIT_END_OF_COLLECTION (1 << 3)   // bit 3: 标志数据采集的结束
#define FLAG_BIT_CRC8_SUPPORT (1 << 7)        // bit 7: 标志启用CRC8校验

// --- 定义结束 ---

/**
 * @brief 自定义事件ID
 * 从EVENT_SDK_END开始，避免与SDK预留事件冲突
 */
typedef enum
{
    EVENT_APP_BASE = EVENT_SDK_END,
    EVENT_ALGO_CONTROL_REQUEST,    /**< 算法启停控制请求 */
    EVENT_PLAY_AUDIO_REQUEST,      /**< 音频播放请求 */
    EVENT_RAW_DATA_ENABLE_REQUEST, /**< 原始数据发送启停请求 */
    EVENT_RAW_AUDIO_READY,         /**< 通知BLE任务：有新的原始语音数据需要发送 */
//    EVENT_ALGO_RESULT_READY,       /**< 通知BLE任务：有新的算法结果需要发送 */
} AppEventId;

/**
 * @brief 音频播放事件参数
 */
typedef enum
{
    AUDIO_ID_START, /**< 播放“开始”音频 */
    AUDIO_ID_STOP,  /**< 播放“结束”音频 */
} AudioId;


/**
 * @brief 用于通过事件传递音频数据的参数结构体
 */
typedef struct
{
    uint8_t *data;        /**< 指向音频数据的指针 */
    uint16_t length;      /**< 数据长度 */
    bool is_first_packet; /**< 是否是数据流的第一个包 */
    bool is_last_packet;  /**< 是否是数据流的最后一个包 */
} AudioDataParam;

/**
 * @brief 用于通过事件传递算法结果的参数结构体
 */
typedef struct
{
    uint8_t *data;   /**< 指向结果数据的指针 */
    uint16_t length; /**< 数据长度 */
} ResultDataParam;

/**
 * @brief 封装并发送原始语音数据
 *
 * 这是暴露给其他任务（如speech_task）调用的主要接口。
 * 它会负责构建符合协议的包头（包括CRC和时间戳），并分片发送数据。
 *
 * @param audio_data 指向纯音频数据的指针
 * @param audio_len 音频数据的长度
 * @param is_first_packet 是否是数据流的第一个包
 * @param is_last_packet 是否是数据流的最后一个包
 * @return int 0表示成功，负数表示错误
 */
int ble_send_raw_audio_packet(const uint8_t *audio_data, uint16_t audio_len, bool is_first_packet, bool is_last_packet);

/**
 * @brief 更新语音原始数据并通知客户端
 *
 * 当原始数据发送使能时，将采集到的语音原始数据通过BLE发送给客户端
 *
 * @param data 原始数据缓冲区
 * @param length 数据长度
 * @return 0: 成功, 其他: 错误码
 */
int speech_service_update_raw_data(const uint8_t *data, uint16_t length);

/**
 * @brief 更新识别结果并通知客户端
 *
 * 当算法运行时，将识别结果通过BLE发送给客户端
 *
 * @param result 识别结果缓冲区
 * @param length 结果长度
 * @return 0: 成功, 其他: 错误码
 */
int speech_service_update_result(const uint8_t *result, uint16_t length);

/**
 * @brief 获取语音原始数据
 *
 * @param data 数据缓冲区
 * @param max_length 缓冲区最大长度
 * @return 实际复制的数据长度
 */
uint16_t speech_service_get_raw_data(uint8_t *data, uint16_t max_length);

/**
 * @brief 获取识别结果
 *
 * @param result 结果缓冲区
 * @param max_length 缓冲区最大长度
 * @return 实际复制的数据长度
 */
uint16_t speech_service_get_result(uint8_t *result, uint16_t max_length);

/**
 * @brief 检查原始数据通知是否已使能
 *
 * @return true: 已使能, false: 未使能
 */
bool speech_service_is_raw_data_notify_enabled(void);

/**
 * @brief 检查识别结果通知是否已使能
 *
 * @return true: 已使能, false: 未使能
 */
bool speech_service_is_result_notify_enabled(void);

/**
 * @brief 初始化语音服务
 *
 * 注册BLE连接回调，必须在BLE协议栈初始化后调用
 */
void speech_service_init(void);

/**
 * @brief BLE application task entry point.
 */
void ble_app_task_entry(void *param);

/**
 * @brief 获取算法控制状态
 *
 * @return 0: 停止, 1: 开始
 */
uint8_t speech_service_get_control_state(void);

/**
 * @brief 获取原始数据发送控制状态
 *
 * @return 1: 发送已开启, 0: 发送已关闭
 */
uint8_t speech_service_get_raw_data_control_state(void);

/**
 * @brief 检查BLE连接状态
 *
 * @return true: 已连接, false: 未连接
 */
bool speech_service_is_connected(void);

extern uint8_t g_ble_raw_pcm_buffer[8000];

#endif /* GALAXY_SDK_BLE_TASK_H_ */
