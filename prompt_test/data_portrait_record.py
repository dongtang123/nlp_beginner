import datetime

# 设置起始日期为2024年7月1日
start_date = datetime.date(2024, 7, 22)
# 设置计划周期
plans = ["无碳日", "低碳日", "高碳日", "无碳日", "低碳日", "高碳日", "放纵餐"]
plans_v = ["维E", "维E", "维B", "维B", "维B", "维B", "维B","维CE", "维CE", "维BC", "维BC", "维BC", "维BC", "维BC"]
plan_colors = ["黄","白"]
# 用户指定需要输出的天数
days_to_show = 90

# 输出日期和计划的对应关系
for i in range(days_to_show):
    current_date = start_date + datetime.timedelta(days=i)
    # 根据周期计算当天的计划
    plan = plans[i % len(plans)]
    plan_color = plan_colors[i%2]
    plan_v = plans_v[i % len(plans_v)]
    print(f"{current_date}: {plan}, {plan_color},{plan_v}")
