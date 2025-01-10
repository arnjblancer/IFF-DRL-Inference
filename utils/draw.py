import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.dates as mdates
def draw_action(cfg):
    return_path = os.path.join('result', cfg.expName, 'returnAgent', 'returnAgent_testAccount.csv')
    risk_path = os.path.join('result', cfg.expName, 'riskAgent', 'riskAgent_testAccount.csv')
    final_path = os.path.join('result', cfg.expName, 'finalAgent', 'finalAgent_testAccount.csv')
    return_data = pd.read_csv(return_path)
    risk_data = pd.read_csv(risk_path)
    final_data = pd.read_csv(final_path)

    # 创建包含四个子图的画布
    fig, (ax2, ax3, ax4) = plt.subplots(3, 1, figsize=(20, 12), sharex=True,
                                        gridspec_kw={'height_ratios': [1, 1, 1]})

    # 绘制主图：股票价格走势

    # 绘制副图1：Return Agent的决策点
    long_points = return_data[return_data['Action'] == 1].index
    short_points = return_data[return_data['Action'] == -1].index
    # close_points = return_data[return_data['Action'] == 2].index

    ax2.plot(return_data['Close'], alpha=0.5)  # 隐藏走势线，通过alpha参数设置透明度为0
    ax2.scatter(long_points, risk_data['Close'][long_points], c='red', s=10, label='Long')
    ax2.scatter(short_points, risk_data['Close'][short_points], c='green', s=10, label='Short')
    # ax2.scatter(close_points, risk_data['Close'][close_points], c='blue', s=10, label='Close')
    # ax2.grid(axis='x')
    ax2.set_ylabel('Price')
    ax2.set_yticks([])
    ax2.set_title('Return Agent Actions')
    ax2.legend()

    # 绘制副图2：Risk Agent的决策点
    long_points = risk_data[risk_data['Action'] == 1].index
    short_points = risk_data[risk_data['Action'] == -1].index
    # close_points = risk_data[risk_data['Action'] == 2].index

    ax3.plot(risk_data['Close'], alpha=0.5)  # 隐藏走势线，通过alpha参数设置透明度为0
    ax3.scatter(long_points, risk_data['Close'][long_points], c='red', s=10, label='Long')
    ax3.scatter(short_points, risk_data['Close'][short_points], c='green', s=10, label='Short')
    # ax3.scatter(close_points, risk_data['Close'][close_points], c='blue', s=10, label='Close')
    ax3.set_title('Risk Agent Actions')
    # ax3.grid(axis='x')
    ax3.set_ylabel('Price')
    ax3.set_yticks([])

    # 绘制副图3：Final Agent的决策点
    long_points = final_data[final_data['Action'] == 1].index
    short_points = final_data[final_data['Action'] == -1].index
    # close_points = final_data[final_data['Action'] == 2].index

    ax4.plot(final_data['Close'], alpha=0.5)  # 隐藏走势线，通过alpha参数设置透明度为0
    ax4.scatter(long_points, risk_data['Close'][long_points], c='red', s=10, label='Long')
    ax4.scatter(short_points, risk_data['Close'][short_points], c='green', s=10, label='Short')
    # ax4.scatter(close_points, risk_data['Close'][close_points], c='blue', s=10,label='Close')
    ax4.set_title('Final Agent Actions')
    # ax4.grid(axis='x')
    ax4.set_xlabel('Days')
    ax4.set_ylabel('Price')
    ax4.set_yticks([])

    plt.xlim(-10, len(final_data['Close']) + 10)

    # 调整子图之间的间距
    plt.subplots_adjust(hspace=0.2)
    save_path = os.path.join('result', cfg.expName, 'actions.jpg')
    plt.savefig(save_path)
    # 显示图形
    plt.show()


def price_plot(dataset):
    dataset = dataset+'.csv'
    csv_file =os.path.join('data', dataset) # 替换为您的CSV文件路径
    data = pd.read_csv(csv_file)
    data['Date'] = pd.to_datetime(data['Date'])

    # 拆分数据为两个部分：2020年底之前和2021年到2022年底
    before_2021 = data[data['Date'] <= '2020-12-31']
    after_2020 = data[data['Date'] >= '2021-01-01']

    # 绘制图像
    fig, ax = plt.subplots(figsize=(12, 6))

    # 绘制2020年底之前的线（橙色）
    ax.plot(before_2021['Date'], before_2021['Close'], color='orange', label='Training Set')

    # 绘制2021年到2022年底的线（蓝色）
    ax.plot(after_2020['Date'], after_2020['Close'], color='blue', label='Test Set')

    # 添加虚线分割线
    ax.axvline(x=pd.to_datetime('2021-01-01'), color='gray', linestyle='--')

    # 设置图像标题和标签
    file_name = os.path.basename(csv_file).split('.')[0].upper()  # 获取文件名并转为大写
    ax.set_title(f'{file_name} Close Price')
    ax.set_xlabel('Date')
    ax.set_ylabel('Closing Price')

    # 添加网格
    ax.grid(True)

    # 设置日期显示格式
    date_format = mdates.DateFormatter('%Y-%m-%d')
    ax.xaxis.set_major_formatter(date_format)

    # 设置刻度
    tick_dates = pd.date_range(start='2007-01-01', end='2022-12-31', periods=8)
    ax.set_xticks(tick_dates)

    # 调整横坐标刻度标签
    plt.xticks(rotation=45, ha='right')

    # 自动调整日期显示格式
    fig.autofmt_xdate()

    # 添加图例

    ax.legend()

    # 调整图表边缘空白
    plt.tight_layout()

    # 保存图片
    output_file = f'{file_name}.jpg'
    plt.savefig(output_file)

    # 显示图像
    plt.show()

def returnRate_plot(dataset):
    name = {'Return Agent': 'returnAgent.csv', 'Risk Agent': 'riskAgent.csv', 'Final Agent': 'finalAgent.csv',
            'B&H': 'B&H.csv',
            'S&H': 'S&H.csv', 'MR': 'MR.csv', 'TF': 'TF.csv', 'DQN-Vanilla': 'DQN-Vanilla.csv',
            'MLP-Vanilla': 'MLP-Vanilla.csv', 'TDQN': 'TDQN.csv'}

    path = 'agent_test_account/'+dataset



    plt.xlabel("Days", fontsize=16)  # 横坐标
    plt.ylabel("Return Rate(%)", fontsize=16)
    for key in name:
        file_path = os.path.join(path, name[key])
        file = pd.read_csv(file_path)
        plt.plot(file['returnRate'], label=key)
    plt.legend()
    plt.grid()
    plt.title('%return rate for test data of {}'.format(dataset), fontsize=16)
    plt.savefig(path+'/{}-returnRate.jpg'.format(dataset), dpi=1000)
    plt.show()

def daily_return_plot(dataset):
    name = {'Return Agent': 'returnAgent.csv', 'Risk Agent': 'riskAgent.csv', 'Final Agent': 'finalAgent.csv'}

    path = 'agent_test_account/'+dataset

    plt.xlabel("Days", fontsize=16)  # 横坐标
    plt.ylabel("Daily Return(%)", fontsize=16)
    for key in name:
        file_path = os.path.join(path, name[key])
        file = pd.read_csv(file_path)
        y_values = file['Returns']
        colors = []
        for i in y_values:
            if i < 0:
                colors.append("red")
            else:
                colors.append("green")
        plt.figure(dpi=500)
        plt.bar(range(len(file['Returns'])), file['Returns'], color=colors)

        plt.grid()
        plt.title("Daily Return of {} in ".format(key)+dataset)
        plt.savefig(path+'/'+dataset+'-daily-retrun-{}.jpg'.format(key), dpi=500)
        plt.show()

