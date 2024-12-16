# fakejob_lili
#招聘市场会受到国内经济的影响，大量的招聘信息中会存在一些虚假描述，如果能够有效鉴别出这些不实信息，对于招聘平台和求职者都具有一定的价值。
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import matplotlib.pyplot as plt

# 配置matplotlib，以便在图表中正确显示中文和负号
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 确保负号显示正常

# 从本地路径读取训练数据集和测试数据集
train_data = pd.read_csv("D:\School\deepstudy\FakeCareers\\fake_job_postings_train.csv")
test_data = pd.read_csv("D:\School\deepstudy\FakeCareers\\fake_job_postings_test.csv")

# 定义包含文本信息的列，这些列将被合并为一个特征向量
text_columns = ['company_profile', 'description', 'requirements', 'benefits']

# 将多个文本列合并为一个长字符串，以便后续处理
# 使用fillna('')处理缺失值，然后使用apply和lambda函数将列合并为一个字符串
train_data['text'] = train_data[text_columns].fillna('').apply(lambda x: ' '.join(x), axis=1)
test_data['text'] = test_data[text_columns].fillna('').apply(lambda x: ' '.join(x), axis=1)

# 使用LabelEncoder对训练数据集中的标签进行编码，将文本标签转换为数字
le = LabelEncoder()
train_data['fraudulent'] = le.fit_transform(train_data['fraudulent'])  # 对fraudulent列进行编码

# 初始化Tokenizer，用于将文本数据转换为模型能理解的数字序列
# 设置num_words=5000，意味着只考虑文本中出现频率最高的5000个单词
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(train_data['text'])  # 基于训练数据集训练Tokenizer

# 将训练数据集和测试数据集的文本转换为数字序列
train_sequences = tokenizer.texts_to_sequences(train_data['text'])
test_sequences = tokenizer.texts_to_sequences(test_data['text'])

# 为了确保所有输入序列长度一致，使用pad_sequences进行填充或截断
max_len = 200  # 设置最大序列长度为200
train_padded = pad_sequences(train_sequences, maxlen=max_len, padding='post')  # 填充训练数据
test_padded = pad_sequences(test_sequences, maxlen=max_len, padding='post')  # 填充测试数据

# 构建一个Sequential模型，这是一个线性堆叠的层次模型
model = Sequential()

# 添加Embedding层，将单词索引映射到高维空间中的向量
model.add(Embedding(input_dim=5000, output_dim=100, input_length=max_len))

# 添加第一个LSTM层，设置单元数为64，并返回序列以供下一个LSTM层使用
model.add(LSTM(64, return_sequences=True))

# 添加Dropout层，丢弃率为0.2，用于减少过拟合
model.add(Dropout(0.2))

# 添加第二个LSTM层，设置单元数为32
model.add(LSTM(32))

# 再次添加Dropout层，进一步减少过拟合
model.add(Dropout(0.2))

# 添加Dense输出层，设置激活函数为sigmoid，用于二分类任务
model.add(Dense(1, activation='sigmoid'))

# 编译模型，设置二分类任务的损失函数、优化器和评估指标
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 设置早停法，监控验证集损失，如果在3个epoch内没有改善则停止训练
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# 设置TensorBoard，用于记录训练过程中的指标，方便后续分析
tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()))

# 训练模型，设置epochs为10，验证集比例为10%，并使用早停法和TensorBoard
history = model.fit(train_padded, train_data['fraudulent'], epochs=10, validation_split=0.1, callbacks=[early_stopping, tensorboard])

# 使用训练好的模型对测试数据进行预测，得到预测结果
y_pred = model.predict(test_padded)

# 将预测结果的多维数组展平，以便后续处理
confidence = y_pred.flatten()

# 创建一个新的DataFrame，包含测试数据集的样本编号和对应的预测置信度
results = pd.DataFrame({
    'job_id': test_data['job_id'],  # 假设测试数据集包含job_id列
    'confidence': confidence  # 预测置信度
})

# 将预测结果保存为CSV文件，不包含索引，使用utf-8编码
results.to_csv('submission02.csv', index=False, encoding='utf-8')

print("预测结果已保存到submission02.csv文件中。")

# 绘制训练和验证过程中的损失和准确率曲线，以便直观地评估模型性能
plt.figure(figsize=(12, 4))

# 绘制损失曲线，比较训练损失和验证损失
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='验证损失')
plt.title('训练和验证损失')
plt.xlabel('Epoch')
plt.ylabel('损失')
plt.legend()

# 绘制准确率曲线，比较训练准确率和验证准确率
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='训练准确率')
plt.plot(history.history['val_accuracy'], label='验证准确率')
plt.title('训练和验证准确率')
plt.xlabel('Epoch')
plt.ylabel('准确率')
plt.legend()

# 调整布局并显示图表
plt.tight_layout()
plt.show()
