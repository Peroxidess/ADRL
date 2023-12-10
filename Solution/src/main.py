import copy
import pandas as pd
import arguments
import os
from sklearn.model_selection import train_test_split
from preprocess import load_data
from preprocess.get_dataset import DataPreprocessing
from preprocess.missing_values_imputation import MVI
from preprocess.representation_learning import RepresentationLearning
from model.ReinforcementLearning import RL

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def run(train_data, test_data, target, args) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    train_data: pd.DataFrame
    test_data: pd.DataFrame
    target: dict
    if args.test_ratio == 0 or not test_data.empty:
        train_set = train_data
        test_set = test_data
    else:
        train_set, test_set = train_test_split(train_data, test_size=args.test_ratio, random_state=args.seed, shuffle=False)

    history_df = pd.DataFrame([])
    train_set_cv, val_set_cv = train_test_split(train_data, test_size=args.val_ratio, random_state=args.seed, shuffle=False)
    test_set_cv = copy.deepcopy(test_set)

    dp = DataPreprocessing(train_set_cv, val_set_cv, test_set_cv, None, seed=args.seed,
                           flag_label_onehot=False,
                           flag_ex_null=True, flag_ex_std_flag=False, flag_ex_occ=False,
                           flag_ca_co_sel=True, flag_ca_fac=True, flag_onehot=True, flag_nor=True,
                           flag_feat_emb=False, flag_RUS=False, flaq_save=False)
    if args.Flag_DataPreprocessing:
        train_set_cv, val_set_cv, test_set_cv, ca_col, co_col, nor = dp.process()

    col_drop = dp.features_ex(train_set_cv) # Drop useless features (high deletion rate, small variance, etc.)
    train_set_cv.drop(columns=col_drop, inplace=True)
    col_ = train_set_cv.columns
    val_set_cv = val_set_cv[col_]
    test_set_cv = test_set_cv[col_]
    ca_col = train_set_cv.filter(regex=r'sparse').columns.tolist()
    co_col = train_set_cv.filter(regex=r'dense').columns.tolist()

    train_label = train_set_cv[[x for x in target.values()]]
    val_label = val_set_cv[[x for x in target.values()]]
    test_label = test_set_cv[[x for x in target.values()]]
    train_x = train_set_cv.drop(columns=target.values())
    val_x = val_set_cv.drop(columns=target.values())
    test_x = test_set_cv.drop(columns=target.values())

    print(f'train_x shape {train_x.shape} | val_x shape {val_x.shape} | test_x shape {test_x.shape}')

    # missing values imputation start
    mvi = MVI(co_col, ca_col, args.seed)
    train_x_filled = mvi.fit_transform(train_x)
    val_x_filled = mvi.transform(val_x)
    test_x_filled = mvi.transform(test_x)
    # missing values imputation end

    # Dimension reduction Start
    represent = RepresentationLearning(dim_input_list=[24, 84, 158])
    train_x_hidden = represent.fit_transform(train_x_filled, val_x=val_x_filled)
    val_x_hidden = represent.transform(val_x_filled)
    test_x_hidden = represent.transform(test_x_filled)
    # Dimension reduction End

    # RL Start
    state_dim = train_x_hidden.shape[1] - 3 - train_x_hidden.filter(regex=r'action').shape[1]
    model = RL(state_dim=state_dim,
               action_dim=train_x_hidden.filter(regex=r'action').shape[1], seed=args.seed, ckpt_dir='', method_RL=args.method_name_RL)

    model.store_data(train_x_hidden, train_label)
    model.agent.learn_class(train_x_hidden.drop(columns=train_x_hidden.filter(regex=r'时间|label').columns), train_label[['label1']],
                            val_x_hidden.drop(columns=val_x_hidden.filter(regex=r'时间|label').columns), val_label[['label1']], episodes_C=200)
    critic_loss_all, actor_loss_all, q_test_list, q_test_df_mean_epoch = model.learn(val_x_hidden, val_label, episodes_RL=300)
    history_df_tmp = pd.DataFrame(critic_loss_all, columns=[f'critic'])
    history_df_tmp[f'action_'] = actor_loss_all
    history_df = pd.concat([history_df, history_df_tmp], axis=1)
    q_test_df = model.eval(copy.deepcopy(test_x_hidden), test_label, type_task='RL')
    # RL End
    return q_test_df, history_df


if __name__ == "__main__":
    args = arguments.get_args()

    for trial in range(args.nrun):
        print('rnum : {}'.format(trial))
        args.seed = (trial * 55) % 2022 + 1 # a different random seed for each run

        # data fetch
        # input: file path
        # output: data with DataFrame
        train_data, test_data, target = load_data.data_load(args.task_name)

        # run model
        # input: train_data
        # output: q_metric, training_history
        q_test_df, history_df = run(train_data, test_data, target, args)
    pass
pass
