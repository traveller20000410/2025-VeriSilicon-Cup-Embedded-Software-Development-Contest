#ifndef GALAXY_SDK_PRECOMPUTED_MELDATA_H_
#define GALAXY_SDK_PRECOMPUTED_MELDATA_H_

#include "constants.h" // 包含此文件以获取 N_MELS 和 N_FFT_MEL 的定义

// 声明预计算的梅尔滤波器组数组
extern const float g_mel_filterbank_data[N_MELS][N_FFT_MEL / 2 + 1];

#endif /* GALAXY_SDK_PRECOMPUTED_DATA_H_ */