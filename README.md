简述： 此目录为基于CNN模型的训练VAD算法模型及SID算法模型的python仿真工程项目。

说明：
train.py：用于训练VAD模型及SID模型的CNN网络结构; \
model.py：存放CNN模型架构； \
label_for_voice.py：通过开源的silero-vad库为音频数据做数据标注，并在output_label文件夹下生成label标签； \
reasoning.py：推理函数，将输入的音频文件输入到模型里进行运算并输出预测的vad及SID结果； \
export_weights_as_header.py：将选定的.pth权重系数文件导出为C可用的.h头文件； \
evaluate_all.py：通过evaluate_wav文件夹里的音频数据来评估模型的性能。\
weights.h：存放训练好的模型权重系数。 \
voice_set/文件夹：存放训练的音频数据。 
model/文件夹：存放训练好的模型。 
output_label/文件夹：存放通过开源的silero-vad库为音频数据做数据标注的label标签。 
evaluatr_wave/文件夹：存放用于的音频文件，这些音频文件未参与模型训练。 
sound_set/文件夹：存放用于训练的音频文件，这些音频文件参与模型训练。
