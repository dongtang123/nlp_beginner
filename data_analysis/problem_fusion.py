import pandas as pd
import os


def problem_1_fusion(name_list, list_1):
    df = {'name': name_list, 'content': list_1}
    df = pd.DataFrame(df)
    df.to_csv('data/problem_01.csv', index=False)


def problem_2_fusion(name_list, list_2):
    df = {'name': name_list, 'content': list_2}
    df = pd.DataFrame(df)
    df.to_csv('data/problem_02.csv', index=False)


def problem_3_fusion(name_list, list_3):
    df = {'name': name_list, 'content': list_3}
    df = pd.DataFrame(df)
    df.to_csv('data/problem_03.csv', index=False)


def problem_4_fusion(name_list, list_4, list_5, list_6, list_7):
    list_problem_4 = []
    for item1, item2, item3, item4 in zip(list_4, list_5, list_6, list_7):
        content = item1 + "[SEP]" + item2 + "[SEP]" + item3 + "[SEP]" + item4
        list_problem_4.append(content)
    df = {'name': name_list, 'content': list_problem_4}
    df = pd.DataFrame(df)
    df.to_csv('data/problem_04.csv', index=False)


def problem_5_fusion(name_list, list_8, list_9, list_10, list_11):
    list_problem_5 = []
    for item1, item2, item3, item4 in zip(list_8, list_9, list_10, list_11):
        content = item1 + "[SEP]" + item2 + "[SEP]" + item3 + "[SEP]" + item4
        list_problem_5.append(content)
    df = {'name': name_list, 'content': list_problem_5}
    df = pd.DataFrame(df)
    df.to_csv('data/problem_05.csv', index=False)


def problem_past_future_fusion(name_list, list_10, list_11):
    list_problem_past_future = []
    for item1, item2 in zip(list_10, list_11):
        content = item1 + "[SEP]" + item2
        list_problem_past_future.append(content)
    df = {'name': name_list, 'content': list_problem_past_future}
    df = pd.DataFrame(df)
    df.to_csv('data/problem_past_future.csv', index=False)


def problem_all(name_list, list_1, list_2, list_3, list_4, list_5, list_6, list_7, list_8, list_9, list_10, list_11):
    list_problem_past_future = []
    for item1, item2, item3, item4, item5, item6, item7, item8, item9, item10, item11 in zip(list_1, list_2, list_3,
                                                                                             list_4, list_5, list_6,
                                                                                             list_7, list_8, list_9,
                                                                                             list_10, list_11):
        content = item1 + "<s>" + item2 + "<s>" + item3 + "<s>" + item4 + "<s>" + item5 + "<s>" + item6 + "<s>" + item7 + "<s>" + item8 + "<s>" + item9 + "<s>" + item10 + "<s>" + item11
        list_problem_past_future.append(content)
    df = {'name': name_list, 'content': list_problem_past_future}
    df = pd.DataFrame(df)
    df.to_csv('data/problem_all.csv', index=False)


def problem_except_2(name_list, list_1, list_3, list_4, list_5, list_6, list_7, list_8, list_9, list_10, list_11):
    list_problem_past_future = []
    for item1, item3, item4, item5, item6, item7, item8, item9, item10, item11 in zip(list_1, list_3,
                                                                                      list_4, list_5, list_6,
                                                                                      list_7, list_8, list_9,
                                                                                      list_10, list_11):
        content = item1 + "<s>" + item3 + "<s>" + item4 + "<s>" + item5 + "<s>" + item6 + "<s>" + item7 + "<s>" + item8 + "<s>" + item9 + "<s>" + item10 + "<s>" + item11
        list_problem_past_future.append(content)
    df = {'name': name_list, 'content': list_problem_past_future}
    df = pd.DataFrame(df)
    df.to_csv('data/problem_except_2.csv', index=False)


def problem_1_5(name_list, list_1, list_8, list_9, list_10, list_11):
    list_1_5 = []
    for item1, item8, item9, item10, item11 in zip(list_1, list_8, list_9, list_10, list_11):
        content = item1 + "<s>" + item8 + "<s>" + item9 + "<s>" + item10 + "<s>" + item11
        list_1_5.append(content)
    df = {'name': name_list, 'content': list_1_5}
    df = pd.DataFrame(df)
    df.to_csv('data/problem_1_5.csv', index=False)


def problem_study(name_list, list_4):
    list_study = []
    for item4 in list_4:
        list_study.append(item4)
    df = {'name': name_list, 'content': list_study}
    df = pd.DataFrame(df)
    df.to_csv('data/problem_study.csv', index=False)


def problem_family(name_list, list_5):
    list_study = []
    for item5 in list_5:
        list_study.append(item5)
    df = {'name': name_list, 'content': list_study}
    df = pd.DataFrame(df)
    df.to_csv('data/problem_family.csv', index=False)


def read_problem(path):
    df = pd.read_csv(path)
    list_1 = []
    list_2 = []
    list_3 = []
    list_4 = []
    list_5 = []
    list_6 = []
    list_7 = []
    list_8 = []
    list_9 = []
    list_10 = []
    list_11 = []
    name_list = []
    for name, content in zip(df['name'], df['content']):
        if name.split('_')[1] == '01':
            name_list.append(name.split('_')[0])
            list_1.append(content)
        elif name.split('_')[1] == '02':
            list_2.append(content)
        elif name.split('_')[1] == '03':
            list_3.append(content)
        elif name.split('_')[1] == '04':
            list_4.append(content)
        elif name.split('_')[1] == '05':
            list_5.append(content)
        elif name.split('_')[1] == '06':
            list_6.append(content)
        elif name.split('_')[1] == '07':
            list_7.append(content)
        elif name.split('_')[1] == '08':
            list_8.append(content)
        elif name.split('_')[1] == '09':
            list_9.append(content)
        elif name.split('_')[1] == '10':
            list_10.append(content)
        elif name.split('_')[1] == '11':
            list_11.append(content)
    return name_list, list_1, list_2, list_3, list_4, list_5, list_6, list_7, list_8, list_9, list_10, list_11


if __name__ == "__main__":
    file_root_path = os.path.join("text.csv")
    name_list, list_1, list_2, list_3, list_4, list_5, list_6, list_7, list_8, list_9, list_10, list_11 = read_problem(
        file_root_path)
    # problem_1_fusion(name_list, list_1)
    # problem_2_fusion(name_list, list_2)
    # problem_3_fusion(name_list, list_3)
    # problem_4_fusion(name_list, list_4, list_5, list_6, list_7)
    # problem_5_fusion(name_list, list_8, list_9, list_10, list_11)
    # problem_past_future_fusion(name_list, list_10, list_11)
    # problem_all(name_list, list_1, list_2, list_3, list_4, list_5, list_6, list_7, list_8, list_9, list_10, list_11)
    # problem_except_2(name_list, list_1, list_3, list_4, list_5, list_6, list_7, list_8, list_9, list_10,
    #                  list_11)
    # problem_1_5(name_list, list_1, list_8, list_9, list_10, list_11)
    problem_study(name_list, list_4)
    problem_family(name_list, list_5)
