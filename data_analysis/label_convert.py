import pandas as pd

import os


def group_information(path):
    df = pd.read_excel(path)
    list_group_information = []
    for item in df['分组信息']:
        if item == "绿色":
            list_group_information.append(0)
        elif item == "黄色":
            list_group_information.append(1)
        elif item == "红色":
            list_group_information.append(2)
    df['分组信息'] = list_group_information
    return df


def bully(dataframe):
    list_verbal_bully = []
    list_physical_bully = []
    list_social_bully = []
    list_cyber_bully = []
    for item in dataframe['言语欺负']:
        if item > 8:
            list_verbal_bully.append(1)  # 1表示受欺负多
        else:
            list_verbal_bully.append(0)  # 0表示受欺负少
    for item in dataframe['身体欺负']:
        if item > 8:
            list_physical_bully.append(1)
        else:
            list_physical_bully.append(0)
    for item in dataframe['社会/关系欺负']:
        if item > 8:
            list_social_bully.append(1)
        else:
            list_social_bully.append(0)
    for item in dataframe['网络欺负']:
        if item > 8:
            list_cyber_bully.append(1)
        else:
            list_cyber_bully.append(0)
    dataframe['言语欺负'] = list_verbal_bully
    dataframe['身体欺负'] = list_physical_bully
    dataframe['社会/关系欺负'] = list_social_bully
    dataframe['网络欺负'] = list_cyber_bully
    return dataframe


def support(dataframe):
    list_subjective_support = []
    list_objective_support = []
    list_utilize_support = []
    for item in dataframe['主观支持']:
        if item <= 12:
            list_subjective_support.append(1)  # 1 主观支持度低
        else:
            list_subjective_support.append(0)  # 0 主观支持度高
    for item in dataframe['客观支持']:
        if item <= 16:
            list_objective_support.append(1)  # 1 客观支持度低
        else:
            list_objective_support.append(0)  # 0 客观支持度高
    for item in dataframe['对支持的利用度']:
        if item <= 14:
            list_utilize_support.append(1)  # 1 对支持的利用低
        else:
            list_utilize_support.append(0)  # 0 对支持的利用高
    dataframe['主观支持'] = list_subjective_support
    dataframe['客观支持'] = list_objective_support
    dataframe['对支持的利用度'] = list_utilize_support
    return dataframe


def self_esteem(dataframe):
    list_self_esteem = []
    for item in dataframe['自尊水平']:
        if item <= 20:
            list_self_esteem.append(2)  # 2 自尊水平低
        elif item > 30:
            list_self_esteem.append(0)  # 0 自尊水平高
        else:
            list_self_esteem.append(1)  # 1 自尊水平中
    dataframe['自尊水平'] = list_self_esteem
    return dataframe


def depression(dataframe):
    list_depression = []
    for item in dataframe['抑郁情绪']:
        if item > 14:
            list_depression.append(2)  # 2 抑郁情绪高
        elif item <= 9:
            list_depression.append(0)  # 0 无抑郁
        else:
            list_depression.append(1)  # 1 轻抑郁
    dataframe['抑郁情绪'] = list_depression
    return dataframe


def school(dataframe):
    list_teacher = []
    list_study = []
    list_classmate = []
    for item in dataframe['师生关系']:
        if item <= 18:
            list_teacher.append(1)  # 1 师生关系待改善
        else:
            list_teacher.append(0)  # 0 师生关系融洽
    for item in dataframe['学业负担']:
        if item <= 10:
            list_study.append(1)  # 1 学业负担大
        else:
            list_study.append(0)  # 0 学业负担小
    for item in dataframe['同学关系']:
        if item <= 14:
            list_classmate.append(1)  # 1 同学关系融洽
        else:
            list_classmate.append(0)  # 0 同学关系待改善
    dataframe['师生关系'] = list_teacher
    dataframe['学业负担'] = list_study
    dataframe['同学关系'] = list_classmate
    return dataframe


def anxiety(dataframe):
    list_anxiety = []
    for item in dataframe['焦虑情绪']:
        if item > 14:
            list_anxiety.append(2)  # 2 较重焦虑
        elif item <= 9:
            list_anxiety.append(0)  # 0 不焦虑
        else:
            list_anxiety.append(1)  # 1 一般焦虑
    dataframe['焦虑情绪'] = list_anxiety
    return dataframe


def family(dataframe):
    list_family = []
    for item in dataframe['家庭功能']:
        if item <= 1.99:
            list_family.append(2)  # 2 家庭功能低
        elif item > 2.99:
            list_family.append(0)  # 0 家庭功能较高
        else:
            list_family.append(1)  # 1 家庭功能中等
    dataframe['家庭功能'] = list_family
    return dataframe


def pressure(dataframe):
    list_pressure = []
    for item in dataframe['压力大']:
        if item > 4:
            list_pressure.append(1)  # 1 压力大
        elif item <= 4:
            list_pressure.append(0)  # 0 无压力
    dataframe['压力大'] = list_pressure
    return dataframe


def life_satisfaction(dataframe):
    list_satisfaction = []
    for item in dataframe['生活满意度']:
        if item >30:
            list_satisfaction.append(0)  # 0 特别满意
        elif 30 >= item >25:
            list_satisfaction.append(1)  # 1 非常满意
        elif 25 >= item >20:
            list_satisfaction.append(2)  # 2 大体满意
        elif 20 >= item >19:
            list_satisfaction.append(3)  # 3 无所谓满意不满意
        elif 19 >= item > 14:
            list_satisfaction.append(4)  # 4 不大满意
        elif 14 >= item > 9:
            list_satisfaction.append(5)  # 4 不满意
        else:
            list_satisfaction.append(6)  # 5 特别不满意
    dataframe['生活满意度'] = list_satisfaction
    return dataframe


def concept(dataframe):
    list_physical_concept = []
    list_appearance_concept = []
    list_sex_concept = []
    for item in dataframe['体能自我概念']:
        if item <= 3:
            list_physical_concept.append(1)  # 1 低
        else:
            list_physical_concept.append(0)  # 0 高
    for item in dataframe['外貌自我概念']:
        if item <= 3:
            list_appearance_concept.append(1)  # 1 低
        else:
            list_appearance_concept.append(0)  # 0 高
    for item in dataframe['异性关系自我概念']:
        if item <= 3:
            list_sex_concept.append(1)  # 1 低
        else:
            list_sex_concept.append(0)  # 0 高
    dataframe['体能自我概念'] = list_physical_concept
    dataframe['外貌自我概念'] = list_appearance_concept
    dataframe['异性关系自我概念'] = list_sex_concept
    return dataframe


def cognitive_expressive(dataframe):
    list_cognitive = []
    list_expressive = []
    for item in dataframe['认知重评']:
        if item <= 18:
            list_cognitive.append(2)  # 2 低
        elif item > 30:
            list_cognitive.append(0)  # 0 高
        else:
            list_cognitive.append(1)  # 1 中等
    for item in dataframe['表达抑制']:
        if item <= 12:
            list_expressive.append(2)  # 2 较少
        elif item > 20:
            list_expressive.append(0)  # 0 较多
        else:
            list_expressive.append(1)  # 1 中等
    dataframe['认知重评'] = list_cognitive
    dataframe['表达抑制'] = list_expressive
    return dataframe


def mental_toughness(dataframe):
    list_toughness = []
    for item in dataframe['心理韧性']:
        if item <= 28:
            list_toughness.append(2)  # 2 低
        elif item > 43:
            list_toughness.append(0)  # 0 高
        else:
            list_toughness.append(1)  # 1 中等
    dataframe['心理韧性'] = list_toughness
    return dataframe


def autonomy_impulse(dataframe):
    list_autonomy = []
    list_impulse = []
    for item in dataframe['自律性']:
        if item <= 6:
            list_autonomy.append(2)  # 2 低
        elif item >= 10:
            list_autonomy.append(0)  # 0 高
        else:
            list_autonomy.append(1)  # 1 中等

    for item in dataframe['冲动控制']:
        if item <= 8:
            list_impulse.append(2)  # 2 低
        elif item >= 13:
            list_impulse.append(0)  # 0 高
        else:
            list_impulse.append(1)  # 1 中等
    dataframe['自律性'] = list_autonomy
    dataframe['冲动控制'] = list_impulse
    return dataframe


if __name__ == "__main__":
    label_path = os.path.join('data/label.xlsx')
    df_1 = group_information(label_path)
    df_2 = bully(df_1)
    df_3 = support(df_2)
    df_4 = self_esteem(df_3)
    df_5 = depression(df_4)
    df_6 = school(df_5)
    df_7 = anxiety(df_6)
    df_8 = family(df_7)
    df_9 = pressure(df_8)
    df_10 = life_satisfaction(df_9)
    df_11 = concept(df_10)
    df_12 = cognitive_expressive(df_11)
    df_13 = mental_toughness(df_12)
    df_14 = autonomy_impulse(df_13)
    df_14.to_csv('label.csv', index=False)
