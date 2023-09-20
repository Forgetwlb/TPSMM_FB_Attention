import pandas as pd
import matplotlib.pyplot as plt

# 读取Excel文件
data = pd.read_excel('baihu.xlsx', sheet_name='2008')
print(data.keys())
# 提取需要绘制的数据列
years = data['2008']
org_counts = data['A']
employee_counts = data['B']

# 创建一个新的图表
plt.figure()

# 绘制体育系统机构数A的趋势图
plt.plot(years, org_counts, label='体育系统机构数A')

# 绘制从业人员数B的趋势图
plt.plot(years, employee_counts, label='从业人员数B')

# 添加标题和标签
plt.title('体育系统机构数A和从业人员数B趋势图')
plt.xlabel('年份')
plt.ylabel('数量')

# 添加图例
plt.legend()

# 显示图表
plt.show()
