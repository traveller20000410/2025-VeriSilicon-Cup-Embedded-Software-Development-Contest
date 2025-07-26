/*
 * audio_task.c
 *
 *  Created on: 2025年7月21日
 *      Author: 廖先强
 */
#include "audio_task.h"

static I2sDevice* i2s_dev_handle = NULL;
static CodecDevice* codec_dev_handle = NULL;

static I2sSubstream i2s_stream_config;
static CodecParams codec_params;


static EventManager audio_event_manager;


// I2S中断回调函数（在DMA模式下，数据发送完成时会调用）
// 对于简单的阻塞式播放，此函数可以为空，但必须提供
static void i2s_irq_handler(const I2sDevice *i2s, uint8_t direction, int size, void *cb_ctx)
{
    // 在连续播放场景下，这里会填充下一个数据缓冲区。
    // 对于单次播放，我们可以暂时留空。
	return;
}

static int audio_hardware_init(void)
{
	int ret;

	codec_dev_handle = hal_codec_get_device(CODEC_ID_0);
	i2s_dev_handle = hal_i2s_get_device(I2S_ID_0);

    if (!codec_dev_handle || !i2s_dev_handle) {
        printf("Error: Failed to get codec or i2s device handle.\n");
        return VPI_ERR_GENERIC;
    }

    // 2. 配置并初始化Codec (WM8960)
    codec_params.sample_rate = 16000;
    codec_params.data_width  = 16;
    codec_params.clk_rate    = 8192000;
    codec_params.fmt         = SND_SOC_DAIFMT_I2S | SND_SOC_DAIFMT_CBS_CFS;
    codec_params.ch_num      = 1;
    codec_params.pll_bypass  = false;

    ret = hal_codec_config(codec_dev_handle, &codec_params);
    if (ret != VPI_SUCCESS) {
        printf("Error: hal_codec_config failed with ret %d\n", ret);
        return ret;
    }

    memset(&i2s_stream_config, 0, sizeof(I2sSubstream));
    i2s_stream_config.direction     = I2S_TRANSMIT; // 设置为发送模式
    i2s_stream_config.sample_rate   = 16000;
    i2s_stream_config.sample_width  = 16;
    i2s_stream_config.channels      = 1;
    i2s_stream_config.work_mode     = DEVICE_MASTER_MODE; // MCU作为I2S主设备
    i2s_stream_config.xfer_mode     = XFER_MODE_DMA; // 使用DMA传输以降低CPU负载
    i2s_stream_config.cb            = i2s_irq_handler;
    i2s_stream_config.cb_context    = NULL;

    ret = hal_i2s_init(i2s_dev_handle);
    if (ret != VSD_SUCCESS) {
        printf("Error: hal_i2s_init failed with ret %d\n", ret);
        hal_codec_close(codec_dev_handle);
        return ret;
    }
    return VPI_SUCCESS;
}

static int audio_event_handler(EventManager manager, EventId event_id, EventParam param)
{
    if (event_id == EVENT_CMD_UPDATE) {
        AppCmd *cmd = (AppCmd *)param;

        // 确认是发给自己的事件
        if (cmd && cmd->target_mgr == EVENT_MGR_AUD) {
            if (cmd->action == APP_CMD_ACTION_START) {
                printf("Playing start sound...\n");
                audio_task_play_pcm(notify_start_raw, notify_start_raw_len);
            } else if (cmd->action == APP_CMD_ACTION_STOP) {
                printf("Playing end sound...\n");
                audio_task_play_pcm(notify_end_raw, notify_end_raw_len);
            }
        }
    }
    return VSD_SUCCESS;
}



void audio_task_entry(void *param)
{
    printf("Audio Task Initialized.\n");
    audio_event_manager = vpi_event_new_manager(
    		EVENT_MGR_AUD,
        audio_event_handler
    );
    if (!audio_event_manager) {
        printf("Failed to create audio event manager!\n");
        return;
    }
    if (audio_hardware_init() != VPI_SUCCESS) {
        printf("Fatal: Audio hardware initialization failed. Audio task will terminate.\n");
        osal_delete_task(NULL); // 初始化失败，删除任务
        return;
    }

    // 任务主循环，等待播放指令
    // 在更复杂的应用中，这里会使用消息队列或事件组来接收播放任务
    if (vpi_event_register(EVENT_CMD_UPDATE, audio_event_manager) != EVENT_OK) {
        printf("Failed to register EVENT_CMD_UPDATE!\n");
        return;
    }
    printf("audio Task listening for ISRevents.\n");
    while(1)
    {
        if (vpi_event_listen(audio_event_manager) != EVENT_OK) {
            printf("audio Task event listen failed!\n");
            return;
        }
    }
}

int audio_task_play_pcm(const uint8_t *data, uint32_t size)
{
    if (!i2s_dev_handle) {
        printf("Error: I2S not initialized.\n");
        return VPI_ERR_DEFAULT;
    }
    if (!data || size == 0) {
        return VPI_ERR_DEFAULT;
    }

    // 配置要播放的数据缓冲区
    // 注意：这里的(void*)data强制转换是为了API兼容性，
    // 在实际DMA使用中，需要确保data指向的内存是DMA可访问的。
    i2s_stream_config.buffer.base   = (void*)data;
    i2s_stream_config.buffer.size   = size;
    i2s_stream_config.buffer.cyclic = false; // 非循环播放

    // 启动I2S传输
    int ret = hal_i2s_start(i2s_dev_handle, &i2s_stream_config);
    if (ret != VSD_SUCCESS) {
        printf("Error: hal_i2s_start failed with ret %d\n", ret);
        return ret;
    }

    // 这是一个简化的阻塞式等待播放完成的逻辑
    // 实际项目中会使用中断/信号量来同步
    uint32_t wait_ms = (size * 1000) / (i2s_stream_config.sample_rate * (i2s_stream_config.sample_width / 8) * i2s_stream_config.channels);
    osal_sleep(wait_ms + 50); // 等待播放完成，并增加一点余量

    hal_i2s_stop(i2s_dev_handle, &i2s_stream_config);
    printf("Playback finished.\n");

    return VPI_SUCCESS;
}

