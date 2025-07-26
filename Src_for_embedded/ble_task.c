/*
 * ble_task.c
 *
 *  Created on: 2025.7.21
 *      Author: ninef
 */

#include <vsbt_config.h>
#include <errno.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "FreeRTOS.h"
#include "task.h"

#include <bluetooth.h>
#include <conn.h>
#include <gatt.h>
#include <uuid.h>
#include "ble_task.h"
#include "vpi_error.h"
#include "hal_crc.h"

#include "osal.h"
#include "uart_printf.h"
#include "vs_conf.h"
#include "vpi_event.h"
#include "vpi_event_def.h"
#include "audio_task.h"
#include "algo_task.h"
#include "constants.h"

static AppCmd algo_cmd_param;
static AppCmd audio_cmd_param;

// 用于存储当前连接协商后的MTU值，默认值为23
static uint16_t current_mtu = 23;

// 全局连接句柄，用于保存当前连接
static struct bt_conn *default_conn;

// 全局采集ID，每次“开始”时递增
static uint8_t collection_id_counter = 0;

static bool raw_data_notify_enabled = false;
static bool result_notify_enabled = false;
static uint8_t control_value = 0;
static uint8_t raw_data_control_value = 0;

// 连接回调声明
static void connected(struct bt_conn *conn, uint8_t err);
static void disconnected(struct bt_conn *conn, uint8_t reason);

#define SPEECH_SERVICE_UUID \
    BT_UUID_DECLARE_128(BT_UUID_128_ENCODE(0x00000001, 0x0003, 0x1000, 0x8000, 0x00805F9B05B5))

#define SPEECH_RAW_DATA_UUID \
    BT_UUID_DECLARE_128(BT_UUID_128_ENCODE(0x00000008, 0x0003, 0x1000, 0x8000, 0x00805F9B05B5))

#define SPEECH_RESULT_UUID \
    BT_UUID_DECLARE_128(BT_UUID_128_ENCODE(0x00000009, 0x0003, 0x1000, 0x8000, 0x00805F9B05B5))

#define CONTROL_UUID \
    BT_UUID_DECLARE_128(BT_UUID_128_ENCODE(0x0000000A, 0x0003, 0x1000, 0x8000, 0x00805F9B05B5))

#define RAW_DATA_UUID \
    BT_UUID_DECLARE_128(BT_UUID_128_ENCODE(0x0000000B, 0x0003, 0x1000, 0x8000, 0x00805F9B05B5))

#define RAW_DATA_MAX_LEN 512
#define RESULT_MAX_LEN 32

static uint8_t speech_raw_data[RAW_DATA_MAX_LEN];
static uint8_t speech_result[RESULT_MAX_LEN];
static uint16_t raw_data_length = 0;
static uint16_t result_length = 0;

// CCC配置回调函数实现
static void speech_raw_data_ccc_cfg_changed(const struct bt_gatt_attr *attr, uint16_t value)
{
    (void)(attr);
    raw_data_notify_enabled = (value == BT_GATT_CCC_NOTIFY);
}

static void speech_result_ccc_cfg_changed(const struct bt_gatt_attr *attr, uint16_t value)
{
    (void)(attr);
    result_notify_enabled = (value & BT_GATT_CCC_NOTIFY) != 0;
}

static ssize_t read_speech_raw_data(struct bt_conn *conn, const struct bt_gatt_attr *attr, void *buf, uint16_t len, uint16_t offset)
{
    return bt_gatt_attr_read(conn, attr, buf, len, offset, speech_raw_data, raw_data_length);
}

static ssize_t read_speech_result(struct bt_conn *conn, const struct bt_gatt_attr *attr, void *buf, uint16_t len, uint16_t offset)
{
    return bt_gatt_attr_read(conn, attr, buf, len, offset, speech_result, result_length);
}

static ssize_t read_control(struct bt_conn *conn, const struct bt_gatt_attr *attr, void *buf, uint16_t len, uint16_t offset)
{
    return bt_gatt_attr_read(conn, attr, buf, len, offset, &control_value, sizeof(control_value));
}

static ssize_t write_control(struct bt_conn *conn, const struct bt_gatt_attr *attr, const void *buf, uint16_t len, uint16_t offset, uint8_t flags)
{
    if (offset != 0)
    {
        return BT_GATT_ERR(BT_ATT_ERR_INVALID_OFFSET);
    }
    if (len != sizeof(control_value))
    {
        return BT_GATT_ERR(BT_ATT_ERR_INVALID_ATTRIBUTE_LEN);
    }

    uint8_t value;
    memcpy(&value, buf, sizeof(value));

    if (value > 1)
    {
        return BT_GATT_ERR(BT_ATT_ERR_VALUE_NOT_ALLOWED);
    }

    if (control_value != value)
    {
        control_value = value;
        
        if (control_value == 1)
        {
            collection_id_counter++;

            // 启动算法
            algo_cmd_param.target_mgr = EVENT_MGR_ALGO;
            algo_cmd_param.action = APP_CMD_ACTION_START;
            vpi_event_notify(EVENT_CMD_UPDATE, &algo_cmd_param);

            // 播放开始音效
            audio_cmd_param.target_mgr = EVENT_MGR_AUD;
            audio_cmd_param.action = APP_CMD_ACTION_START;
            vpi_event_notify(EVENT_CMD_UPDATE, &audio_cmd_param);
        }
        else
        {
//            // 停止算法
            algo_cmd_param.target_mgr = EVENT_MGR_ALGO;
            algo_cmd_param.action = APP_CMD_ACTION_STOP;
            vpi_event_notify(EVENT_CMD_UPDATE, &algo_cmd_param);
            
            // 播放结束音效
            audio_cmd_param.target_mgr = EVENT_MGR_AUD;
            audio_cmd_param.action = APP_CMD_ACTION_STOP;
            vpi_event_notify(EVENT_CMD_UPDATE, &audio_cmd_param);
        }
    }

    return len;
}

static ssize_t read_raw_data_control(struct bt_conn *conn, const struct bt_gatt_attr *attr, void *buf, uint16_t len, uint16_t offset)
{
    return bt_gatt_attr_read(conn, attr, buf, len, offset, &raw_data_control_value, sizeof(raw_data_control_value));
}

static ssize_t write_raw_data_control(struct bt_conn *conn, const struct bt_gatt_attr *attr, const void *buf, uint16_t len, uint16_t offset, uint8_t flags)
{
    if (offset != 0)
    {
        return BT_GATT_ERR(BT_ATT_ERR_INVALID_OFFSET);
    }
    if (len != sizeof(raw_data_control_value))
    {
        return BT_GATT_ERR(BT_ATT_ERR_INVALID_ATTRIBUTE_LEN);
    }

    uint8_t value;
    memcpy(&value, buf, sizeof(value));

    if (value > 1)
    {
        return BT_GATT_ERR(BT_ATT_ERR_VALUE_NOT_ALLOWED);
    }

    if (raw_data_control_value != value)
    {
        raw_data_control_value = value;
    }

    return len;
}

static struct bt_conn_cb conn_callbacks = {
    .connected = connected,
    .disconnected = disconnected,
};

static void connected(struct bt_conn *conn, uint8_t err)
{
    if (err)
    {
        current_mtu = 23;
        return;
    }
    default_conn = bt_conn_ref(conn);
    current_mtu = bt_gatt_get_mtu(default_conn);
    uart_printf("Connection established. Negotiated MTU is %d bytes.\r\n", current_mtu);
}

static void disconnected(struct bt_conn *conn, uint8_t reason)
{
    if (default_conn)
    {
        bt_conn_unref(default_conn);
        default_conn = NULL;
    }
    current_mtu = 23;
    uart_printf("Disconnected. MTU reset to default.\r\n");
}

static struct bt_gatt_attr speech_attrs[] = {
    BT_GATT_PRIMARY_SERVICE(SPEECH_SERVICE_UUID),
    BT_GATT_CHARACTERISTIC(SPEECH_RAW_DATA_UUID, BT_GATT_CHRC_READ | BT_GATT_CHRC_NOTIFY, BT_GATT_PERM_READ, read_speech_raw_data, NULL, NULL),
    BT_GATT_CCC(speech_raw_data_ccc_cfg_changed, BT_GATT_PERM_READ | BT_GATT_PERM_WRITE),
    BT_GATT_CHARACTERISTIC(SPEECH_RESULT_UUID, BT_GATT_CHRC_READ | BT_GATT_CHRC_NOTIFY, BT_GATT_PERM_READ, read_speech_result, NULL, NULL),
    BT_GATT_CCC(speech_result_ccc_cfg_changed, BT_GATT_PERM_READ | BT_GATT_PERM_WRITE),
    BT_GATT_CHARACTERISTIC(CONTROL_UUID, BT_GATT_CHRC_READ | BT_GATT_CHRC_WRITE, BT_GATT_PERM_READ | BT_GATT_PERM_WRITE, read_control, write_control, NULL),
    BT_GATT_CHARACTERISTIC(RAW_DATA_UUID, BT_GATT_CHRC_READ | BT_GATT_CHRC_WRITE, BT_GATT_PERM_READ | BT_GATT_PERM_WRITE, read_raw_data_control, write_raw_data_control, NULL),
};

static struct bt_gatt_service speech_recognition_svc = BT_GATT_SERVICE(speech_attrs);

void speech_service_init(void)
{
    bt_conn_cb_register(&conn_callbacks);
    bt_gatt_service_register(&speech_recognition_svc);
}

#pragma pack(1)
typedef struct
{
    uint8_t block_type;
    uint8_t crc8_rsv;
    uint8_t sensor_type;
    uint8_t data_subtype;
    uint8_t version;
    uint8_t data_format;
    uint8_t flag;
    uint8_t collection_id;
    uint16_t sample_count;
    uint16_t sample_rate;
    uint32_t timestamp_low32;
    uint16_t timestamp_high16;
    uint16_t data_length;
} SpeechDataPacketHeader;
#pragma pack()
static int flag = 0;
int ble_send_raw_audio_packet(const uint8_t *audio_data, uint16_t audio_len, bool is_first_packet, bool is_last_packet)
{
    if (!audio_data || audio_len == 0)
        return -1;
    if (!raw_data_control_value)
        return 0;
    if (!flag) {
    	flag = 1;
        current_mtu = ble_get_cur_mtu();
    }

    //uart_printf(" Negotiated MTU is %d bytes.\r\n", current_mtu);
    // 1. 计算每个BLE包最大能容纳的音频数据长度
    const uint16_t max_ble_payload = current_mtu > 3 ? current_mtu - 3 : 20;
    const uint16_t header_size = sizeof(SpeechDataPacketHeader);

    if (max_ble_payload <= header_size) {
        uart_printf("Error: MTU size too small for data packet header.\r\n");
        return -1;
    }
    const uint16_t max_audio_chunk_len = max_ble_payload - header_size;

    // 2. 使用静态缓冲区来构造每个要发送的小包
    static uint8_t packet_chunk_buffer[512]; // 假设MTU不超过512

    const uint8_t *audio_data_ptr = audio_data;
    uint16_t remaining_audio_len = audio_len;
    bool is_first_chunk = true;

    // 获取一次时间戳，用于整个数据块的所有子包
    const uint64_t current_utc_ms = (uint64_t)xTaskGetTickCount() * portTICK_PERIOD_MS;

    // 3. 循环发送，直到所有音频数据都已分包
    while (remaining_audio_len > 0)
    {
        // 3.1 计算当前这个小包要发送的音频数据长度
        const uint16_t current_audio_chunk_len = (remaining_audio_len > max_audio_chunk_len) ? max_audio_chunk_len : remaining_audio_len;

        // 3.2 构造当前小包的包头
        SpeechDataPacketHeader header;
        header.block_type = BLOCK_TYPE_RAW_DATA;
        header.sensor_type = SENSOR_TYPE_MIC;
        header.data_subtype = DATA_SUBTYPE_MIC_RAW;
        header.version = 0x00;
        header.data_format = (DATA_FORMAT_UNITS_PER_SAMPLE_1 << 4) | DATA_FORMAT_TYPE_INT16;
        header.collection_id = collection_id_counter; // 同一采集过程的ID相同

        // 设置标志位
        header.flag = 0;
        if (is_first_chunk && is_first_packet) {
            header.flag |= FLAG_BIT_START_OF_COLLECTION;
        }
        const bool is_last_chunk = (current_audio_chunk_len == remaining_audio_len);
        if (is_last_chunk && is_last_packet) {
            header.flag |= FLAG_BIT_END_OF_COLLECTION;
        }
        header.flag |= FLAG_BIT_CRC8_SUPPORT;

        header.sample_count = current_audio_chunk_len / 2; // 当前小包的采样点数
        header.sample_rate = 16000;
        header.timestamp_low32 = (uint32_t)current_utc_ms;
        header.timestamp_high16 = (uint16_t)(current_utc_ms >> 32);
        header.data_length = current_audio_chunk_len; // 当前小包的音频数据长度

        // 为当前小包的音频数据计算CRC
        CrcInput crc_input;
        uint32_t crc_result;
        crc_input.p_buffer = (void *)audio_data_ptr;
        crc_input.buffer_length = current_audio_chunk_len;
        crc_input.poly = CRC_POLYNOMIAL_CRC_8;
        crc_input.cal_switch = CRC_LMS_MSB;
        if (hal_crc_calculate(&crc_input, &crc_result) == VSD_SUCCESS) {
            header.crc8_rsv = (uint8_t)crc_result;
        } else {
            header.crc8_rsv = 0x00;
        }

        // 3.3 组装小包 (包头 + 音频数据块)
        const uint16_t total_chunk_size = header_size + current_audio_chunk_len;
        if (total_chunk_size > sizeof(packet_chunk_buffer)) {
             uart_printf("Error: Chunk size exceeds static buffer.\r\n");
             return -1; // 安全检查
        }
        memcpy(packet_chunk_buffer, &header, header_size);
        memcpy(packet_chunk_buffer + header_size, audio_data_ptr, current_audio_chunk_len);
        osal_sleep(1);
        // 3.4 发送组装好的小包
        int err = speech_service_update_raw_data(packet_chunk_buffer, total_chunk_size);
        if (err)
        {
            uart_printf("Failed to send chunk, error: %d\r\n", err);
            return err;
        }

        // 3.5 更新指针和剩余长度，为下一个循环做准备
        remaining_audio_len -= current_audio_chunk_len;
        audio_data_ptr += current_audio_chunk_len;
        is_first_chunk = false;
    }

    return 0;
}

int speech_service_update_raw_data(const uint8_t *data, uint16_t length)
{
    int err = 0;
    if (!default_conn)
        return -ENOTCONN;
    if (!data || length == 0)
        return -EINVAL;

    uint16_t copy_len = (length < sizeof(speech_raw_data)) ? length : sizeof(speech_raw_data);
    memcpy(speech_raw_data, data, copy_len);
    raw_data_length = copy_len;

    if (raw_data_notify_enabled)
    {
        const struct bt_gatt_attr *attr = &speech_recognition_svc.attrs[2];
        err = bt_gatt_notify(default_conn, attr, data, length);
        if (err)
        {
            return (err == -ENOTCONN) ? 0 : err;
        }
    }
    return 0;
}

int speech_service_update_result(const uint8_t *result, uint16_t length)
{
    int err = 0;
    if (!result || length == 0 || length > sizeof(speech_result))
        return -EINVAL;
    if (!control_value)
        return 0;

    memcpy(speech_result, result, length);
    result_length = length;

    const struct bt_gatt_attr *attr = &speech_recognition_svc.attrs[5];
    if (result_notify_enabled)
    {
        err = bt_gatt_notify(default_conn, attr, speech_result, result_length);
    }
    if (err)
    {
        return (err == -ENOTCONN) ? 0 : err;
    }
    return 0;
}

uint16_t speech_service_get_raw_data(uint8_t *data, uint16_t max_length)
{
    if (!data || max_length == 0)
        return 0;
    uint16_t copy_length = (raw_data_length < max_length) ? raw_data_length : max_length;
    memcpy(data, speech_raw_data, copy_length);
    return copy_length;
}

uint16_t speech_service_get_result(uint8_t *result, uint16_t max_length)
{
    if (!result || max_length == 0)
        return 0;
    uint16_t copy_length = (result_length < max_length) ? result_length : max_length;
    memcpy(result, speech_result, copy_length);
    return copy_length;
}

bool speech_service_is_raw_data_notify_enabled(void)
{
    return raw_data_notify_enabled;
}

bool speech_service_is_result_notify_enabled(void)
{
    return result_notify_enabled;
}

uint8_t speech_service_get_raw_data_control_state(void)
{
    return raw_data_control_value;
}

bool speech_service_is_connected(void)
{
    return default_conn != NULL;
}

uint8_t speech_service_get_control_state(void)
{
    return control_value;
}

static int ble_event_handler(EventManager manager, EventId event_id, EventParam param)
{
    if (event_id == EVT_RAW_PCM_FOR_BLE) {
        if (raw_data_control_value == 1) { // 仅当开关打开时发送
        	ble_send_raw_audio_packet(g_ble_raw_pcm_buffer, PDM_BUFFER_SIZE_BYTES, false, false);
        }
    }
    if (event_id == EVENT_ALGO_RESULT_READY) {
        ResultDataParam *result = (ResultDataParam *)param;
        if (result && result->data && result->length > 0) {
            // 使用同事留下的函数发送结果
            speech_service_update_result(result->data, result->length);
        }
    }
    return VSD_SUCCESS;
}


void initialize_crc_module(void)
{
    // 1. 获取CRC设备句柄 (SDK中似乎没有这个函数，init可能直接用)
     CrcDevice* crc_dev = hal_crc_get_device();
     if (!crc_dev) {
         printf("Failed to get CRC device!\n");
         return;
     }

    // 2. 初始化CRC模块
    // 根据手册9.4.2节，只需要调用init即可
    int ret = hal_crc_init(); // 假设函数原型是这样
    if (ret != VSD_SUCCESS) {
        printf("hal_crc_init failed with ret %d\n", ret);
    } else {
        printf("CRC module initialized successfully.\n");
    }
}

void ble_app_task_entry(void *param)
{
    EventManager ble_event_manager = vpi_event_new_manager(EVENT_MGR_BLE, ble_event_handler);
    if (!ble_event_manager) {
        printf("Failed to create ble event manager!\n");
        return;
    }

    if (vpi_event_register(EVT_RAW_PCM_FOR_BLE, ble_event_manager) != EVENT_OK) {
        printf("Failed to register EVT_RAW_PCM_FOR_BLE!\n");
        return;
    }
    if (vpi_event_register(EVENT_ALGO_RESULT_READY, ble_event_manager) != EVENT_OK) {
        printf("Failed to register EVENT_ALGO_RESULT_READY!\n");
        return;
    }
    initialize_crc_module();
    printf("BLE Task listening for events.\n");
    while (1)
    {
        if (vpi_event_listen(ble_event_manager) != EVENT_OK) {
            printf("BLE Task event listen failed!\n");
            // 在实际应用中，这里可能需要错误处理和恢复逻辑
        }
    }
}
