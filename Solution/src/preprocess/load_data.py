import itertools
import numpy as np
import pandas as pd
import re


def data_load(task_name):
    test_data = pd.DataFrame([])
    if 'liver' in task_name:
        if 'processed' in task_name:
            file_name_tra = '../DataSet/LiverCancerAblation/train_data_processed_lv2.csv'
            data = pd.read_csv(file_name_tra, encoding='gb18030')
            data.dropna(how='all', axis=1, thresh=20, inplace=True)
            col_test = data.filter(regex=r'检查结果|消融范围').columns
            data[col_test] = data[col_test].replace(regex={'<': '', '《': '', '小于': '', '＜': '', '〈': '', 'E ': 'E', 'x10  ': 'E+', '>': '', '\.\.': '\.'})
            data.replace({'..': '.'}, inplace=True)
            data[col_test] = data[col_test].apply(pd.to_numeric, **{'errors': 'coerce'})

            data_ = data.drop(columns=data.filter(regex=r'随访').columns)
            col_focus_id = data.filter(regex=r'治疗_治疗信息_治疗信息_1_病灶信息_病灶编号').columns.values[0]
            data_.dropna(subset=[col_focus_id], inplace=True)

            col_ablation = data_.filter(regex=r'消融范围').columns
            data_.loc[:, col_ablation] = data_.loc[:, col_ablation].replace({0: np.nan})
            data_.sort_values(by=['病例编号', '次序_lv1', '次序_lv2'], inplace=True)

            index_once = pd.Index([])
            id_unique = data_['病例编号'].unique()
            for id_ in id_unique:
                data_single = data_[data_['病例编号'] == id_]
                col_lv = data_single.filter(regex=r'次序').columns
                for e in itertools.product(set(data_single[col_lv[0]]), set(data_single[col_lv[1]])):
                    data_single_ = data_single[(data_single[col_lv[0]] == e[0]) & (data_single[col_lv[1]] == e[1])]
                    data_single_lv3 = data_single_.drop(columns=data_single.filter(regex=r'次序|编号').columns)
                    if data_single_lv3.shape[0] == 1:
                        index_once = index_once.append(data_single_lv3.index)

            data_.drop(columns=data_.filter(regex=r'病灶编号').columns, inplace=True)
            test_data = data_.loc[index_once]
            train_data = data_.drop(index=test_data.index)

            col_tmp = train_data.filter(regex=r'次序|病灶编号').columns
            train_data[col_tmp] = train_data[col_tmp].astype('object')
        else:
            file_name_tra = '../DataSet/LiverCancerAblation/row_dataset.csv'
            data = pd.read_csv(file_name_tra).dropna(how='all', axis=1)
            data_ = data.drop(columns=data.filter(regex=r'特殊CRF|分中心名称|-ID|-姓名|联系人|联系电话').columns)
            data_hospital = data_[data_['中心名称'] == '301医院']

            list_col_tmp = data_hospital.filter(regex=r'-\d-').columns

            dict_name_tmp = {}
            for col_ in list_col_tmp:
                col_str = col_.split('-')
                col_new = '-'.join(col_str[:3]) + f'-{col_str[-1]}-' + '-'.join(col_str[4:6]) + f'-{col_str[3]}'
                dict_name_tmp[col_] = col_new
            data_hospital.rename(columns=dict_name_tmp, inplace=True)
            data_ = pd.DataFrame([])
            for i in range(1, 8):
                data_single0 = data_hospital.filter(regex=r'病例编号')
                data_single0['次序_lv1'] = i
                data_single1 = data_hospital.filter(regex=r'基线')
                data_single2 = data_hospital.filter(regex=rf'治疗{i}')
                data_single3 = data_hospital.filter(regex=rf'随访{i}')
                list_col = data_single2.columns.append(data_single3.columns)
                dict_name = {}
                for col_ in list_col:
                    col_str = col_.split('-')
                    col_str[0] = col_str[0][:-1]
                    col_new = '_'.join(col_str)
                    dict_name[col_] = col_new
                data_single2.rename(columns=dict_name, inplace=True)
                data_single3.rename(columns=dict_name, inplace=True)
                data_singlelv2 = pd.DataFrame([])
                for j in range(1, 35):
                    data_single0['次序_lv2'] = j
                    data_test = data_single2.filter(regex=r'术前化验|术后化验')
                    data_single2.drop(columns=data_test.columns, inplace=True)
                    data_singlelv2_2 = data_single2.filter(regex=rf'_{j}\b')
                    data_singlelv2_3 = data_single3.filter(regex=rf'_{j}\b')
                    if data_singlelv2_2.empty and data_singlelv2_3.empty:
                        print(f'j {j}')
                        break
                    list_col_lv2 = data_singlelv2_2.columns.append(data_singlelv2_3.columns)
                    dict_name_lv2 = {}
                    for col_ in list_col_lv2:
                        col_str = col_.split('_')
                        col_new = '_'.join(col_str[:-1])
                        dict_name_lv2[col_] = col_new
                    data_singlelv2_2.rename(columns=dict_name_lv2, inplace=True)
                    data_singlelv2_3.rename(columns=dict_name_lv2, inplace=True)

                    col_lv3 = data_singlelv2_2.filter(regex=rf'\d\b')
                    col_ = data_singlelv2_2.columns.drop(col_lv3)
                    data_singlelv2_2_ = data_singlelv2_2[col_]
                    for n in range(1, 20):

                        data_singlelv3_2 = data_singlelv2_2.filter(regex=rf'\D{n}\b')
                        if data_singlelv3_2.empty:
                            print(f'n {n}')
                            break
                        list_col_lv3 = data_singlelv3_2.columns
                        dict_name_lv3 = {}
                        for col_ in list_col_lv3:
                            col_str = col_.split('_')
                            if str(n) in col_str:
                                col_tmp = col_str[0:3]
                                col_tmp.extend(col_str[4:6])
                                col_new = '_'.join(col_tmp)
                            elif str(n) in col_str[-1]:
                                col_tmp = col_str[-1]
                                col_str[-1] = ''.join([i for i in col_tmp if not i.isdigit()])
                                col_new = '_'.join(col_str)
                            dict_name_lv3[col_] = col_new
                        data_singlelv3_2.rename(columns=dict_name_lv3, inplace=True)
                        data_singlelv3_2['次序_lv3'] = n
                        data_singlelv2_ = pd.concat([data_single0, data_single1, data_singlelv2_2_, data_singlelv3_2, data_test, data_singlelv2_3], axis=1)
                        data_singlelv2 = pd.concat([data_singlelv2, data_singlelv2_], axis=0)

                for index_ in set(data_singlelv2.index):
                    data_singlelv2.loc[index_, ['随访_随访信息_随访_完全消融（术后一个月评判）',
                                                '随访_随访信息_随访_局部进展',
                                                '随访_随访信息_随访_肝外转移',
                                                '随访_随访信息_随访_肝内远处转移']] = \
                        data_singlelv2.loc[index_][['随访_随访信息_随访_完全消融（术后一个月评判）',
                                                              '随访_随访信息_随访_局部进展',
                                                              '随访_随访信息_随访_肝外转移',
                                                              '随访_随访信息_随访_肝内远处转移']].fillna(method='ffill')
                    pass
                col1 = set(data_singlelv2.columns)
                data_ = pd.concat([data_, data_singlelv2], axis=0)
                col2 = set(data_.columns)

            train_data = data_.dropna(axis=0, thresh=10)# sample
            train_data.dropna(axis=1, thresh=10, inplace=True)# features
            col_time = train_data.filter(regex=r'治疗信息_时间').columns
            train_data.dropna(subset=[col_time[0]], inplace=True)
            col_method = train_data.filter(regex=r'治疗信息_治疗方式').columns[0]
            train_data = train_data[train_data[col_method] == '微波']

            train_data.to_csv('./train_data_lv2.csv', encoding='gb18030', index=False)
            col_drop = train_data.filter(regex=r'合并症|备注|药物治疗时间|药物名称|其他补充|退针信息|检查项目|单位').columns
            train_data.drop(columns=col_drop, inplace=True)

            train_data['relapse'] = 0
            train_data['ablation'] = 1
            col_tmp = train_data.filter(regex=r'转移|局部进展|完全消融').columns
            for col_ in col_tmp:
                train_data['relapse'][train_data[col_] == '转移'] = 1
                train_data['relapse'][train_data[col_] == '有'] = 1
                train_data['ablation'][train_data[col_] == '否'] = 0
                train_data['ablation'][train_data[col_] == '有'] = 0
            train_data.sort_values(by=['病例编号', '次序_lv1', '次序_lv2', '次序_lv3'], inplace=True)

            def data_treatment_ave2(dttrmt):
                if not dttrmt:
                    return dttrmt
                if not ((type(dttrmt) == float) or (type(dttrmt) == int)):
                    nums = [int(s) for s in re.split(r'-|\.|,|/|\+|、', dttrmt) if s != '']
                    return float(sum(nums) / len(nums))
                else:
                    return float(dttrmt)

            col_tmp = train_data.filter(regex=r'功率').columns
            train_data[col_tmp] = train_data[col_tmp].applymap(data_treatment_ave2)
            col_tmp = train_data.filter(regex=r'治疗信息_时间').columns
            train_data[col_tmp] = train_data[col_tmp].applymap(data_treatment_ave2)
            col_drop = train_data.filter(regex=r'日期').columns
            train_data.drop(columns=col_drop, inplace=True)
            train_data.to_csv('./train_data_processed_lv2.csv', index=False, encoding='gb18030')

        train_data.drop(columns=train_data.filter(regex=r'肝外转移|转移器官|入标本库的时间|针|消融穿刺').columns, inplace=True)
        test_data.drop(columns=test_data.filter(regex=r'肝外转移|转移器官|入标本库的时间|针|消融穿刺').columns, inplace=True)
        col_action = train_data.filter(regex=r'时间').columns[0]
        train_data.rename(columns={'ablation': 'label2', 'relapse': 'label1', col_action: 'action'}, inplace=True)
        test_data.rename(columns={'ablation': 'label2', 'relapse': 'label1', col_action: 'action'}, inplace=True)
        target_dict = {'label1': 'label1', 'label2': 'label2'}
    return train_data, test_data, target_dict
