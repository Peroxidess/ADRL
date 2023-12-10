import copy
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from model.ae import AutoEncoder
import torch
import torch.optim as optim
import torch.utils.data as Data


class RepresentationLearning:
    def __init__(self, dim_input_list, seed=2022):
        self.model_lv1 = AE(dim_input_list[0], dim_input_list[0] // 6, seed)
        self.model_lv2 = AE(dim_input_list[1],  dim_input_list[1] // 6, seed)
        self.model_lv3 = AE(dim_input_list[2], dim_input_list[2] // 6, seed)

    def fit_transform(self, train_x, val_x=None):
        col_ds = train_x.filter(regex=r'dense|sparse').columns
        col_lv1 = train_x.filter(regex=r'基线').columns
        data_x_lv1, data_z_lv1, data_val_x_lv1, data_val_z_lv1 = self.model_lv1.fit_transform(train_x[col_lv1],
                                                                                              val_x[col_lv1])

        col_tum = train_x.filter(regex=r'肿瘤信息').columns
        col_pre = train_x.filter(regex=r'术前').columns

        tra_lv2 = pd.concat([data_z_lv1, train_x[col_tum.append(col_pre)]], axis=1)
        val_lv2 = pd.concat([data_val_z_lv1, val_x[col_tum.append(col_pre)]], axis=1)
        data_x_lv2, data_z_lv2, data_val_x_lv2, data_val_z_lv2 = self.model_lv2.fit_transform(tra_lv2, val_lv2)

        col_tx = train_x.filter(regex=r'治疗信息').columns

        tra_lv3 = pd.concat([data_z_lv2, train_x[col_tx]], axis=1)
        val_lv3 = pd.concat([data_val_z_lv2, val_x[col_tx]], axis=1)

        data_x_lv3, data_z_lv3, data_val_x_lv3, data_val_z_lv3 = self.model_lv3.fit_transform(tra_lv3, val_lv3)
        col_sp = train_x.columns.drop(col_ds)
        data_x_all = pd.concat([data_x_lv3, train_x[col_tx], train_x[col_sp]], axis=1)
        return data_x_all

    def transform(self, x):
        col_ds = x.filter(regex=r'dense|sparse').columns
        col_lv1 = x.filter(regex=r'基线').columns
        data_x_lv1, data_z_lv1 = self.model_lv1.transform(x[col_lv1])

        col_tum = x.filter(regex=r'肿瘤信息').columns
        col_pre = x.filter(regex=r'术前').columns

        tra_lv2 = pd.concat([data_z_lv1, x[col_tum.append(col_pre)]], axis=1)
        data_x_lv2, data_z_lv2 = self.model_lv2.transform(tra_lv2)

        col_tx = x.filter(regex=r'治疗信息').columns

        tra_lv3 = pd.concat([data_z_lv2, x[col_tx]], axis=1)

        data_x_lv3, data_z_lv3 = self.model_lv3.transform(tra_lv3)
        col_sp = x.columns.drop(col_ds)
        data_x_all = pd.concat([data_x_lv3, x[col_tx], x[col_sp]], axis=1)
        return data_x_all


class AE():
    def __init__(self, dim_input, z_dim, seed):
        self.seed = seed
        self.dim_input = dim_input
        self.model = AutoEncoder(dim_input, z_dim=z_dim, seed=seed)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def input_process(self, data, y):
        data_dataloader = self.MyDataset(data, y)
        return data_dataloader

    def transform(self, x):
        x_rec, z_ = self.model(torch.Tensor(x.values))
        x_np = x_rec.detach().cpu().numpy()
        z_np = z_.detach().cpu().numpy()
        mse_re, mae_re, mape_re, r2_re = metric_rec(x.values, x_np)
        print(f'mae_re {mae_re} mape_re {mape_re}')
        x_df = pd.DataFrame(x_np, columns=x.columns, index=x.index)
        z_df = pd.DataFrame(z_np, index=x.index)
        z_df.columns = [str(col) + '_dense' for col in range(z_df.shape[1])]
        return x_df, z_df

    def fit_transform(self, x, val_x=None, epoch=60):
        data_dataloader = self.input_process(x, None)
        optim_ = optim.Adam(self.model.parameters(), lr=5e-3, weight_decay=1e-5)
        data_loader = Data.DataLoader(data_dataloader, batch_size=256, worker_init_fn=np.random.seed(self.seed))
        for iter_count in range(epoch):
            total_vae_loss_ = 0
            self.model.train()
            for data in data_loader:
                data = data.to(self.device)
                noise = torch.randn(size=data.shape).to(self.device)
                optim_.zero_grad()
                x_rec, z = self.model(data + 1e-2 * noise)
                rec_loss = self.model.loss(data, x_rec)
                rec_loss.backward()
                optim_.step()
            total_vae_loss_ += rec_loss.detach().cpu().numpy()
            if iter_count % 5 == 0:
                self.model.eval()
                x_tensor = torch.Tensor(val_x.values)
                x_hr, z_hr = self.model(x_tensor)
                self.model.train()
                x_np = x_hr.detach().cpu().numpy()
                z_np = z_hr.detach().cpu().numpy()
                mse_re, mae_re, mape_re, r2_re = metric_rec(val_x.values, x_np)
                print(f'mae_re {mae_re} mape_re {mape_re}')
        self.model.eval()
        x_tensor = torch.Tensor(x.values)
        x_hr, z_hr, _, _ = self.model(x_tensor)
        x_np = x_hr.detach().cpu().numpy()
        z_np = z_hr.detach().cpu().numpy()
        x_df = pd.DataFrame(x_np, columns=x.columns, index=x.index)
        z_df = pd.DataFrame(z_np, index=x.index)
        z_df.columns = [str(col) + '_dense' for col in range(z_df.shape[1])]

        x_tensor_val = torch.Tensor(val_x.values)
        x_hr_val, z_hr_val, _, _ = self.model(x_tensor_val)
        x_np_val = x_hr_val.detach().cpu().numpy()
        z_np_val = z_hr_val.detach().cpu().numpy()
        x_val_df = pd.DataFrame(x_np_val, columns=val_x.columns, index=val_x.index)
        z_val_df = pd.DataFrame(z_np_val, index=val_x.index)
        z_val_df.columns = [str(col) + '_dense' for col in range(z_val_df.shape[1])]
        return x_df, z_df, x_val_df, z_val_df

    class MyDataset(Data.Dataset):
        def __init__(self,
                     data,
                     label=None,
                     random_seed=0):
            self.rnd = np.random.RandomState(random_seed)
            data = data.astype('float32')

            list_data = []
            if label is not None:
                for index_, values_ in data.iterrows():
                    y = torch.LongTensor([label.loc[index_].astype('int64')]).squeeze()
                    x = data.loc[index_].values
                    list_data.append((x, y))
            else:
                for index_, values_ in data.iterrows():
                    x = data.loc[index_].values
                    list_data.append((x))

            self.shape = x.shape
            self.data = list_data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            data = self.data[idx]
            return data


def metric_rec(data, data_rec):
    mms = MinMaxScaler(feature_range=(0.1, 1))
    data_nor = mms.fit_transform(data)
    data_rec_nor = mms.transform(data_rec)
    mse_re = mean_squared_error(data, data_rec)
    mae_re = mean_absolute_error(data, data_rec)
    mape_re = mean_absolute_percentage_error(data_nor, data_rec_nor)
    r2_re = r2_score(data, data_rec)
    return mse_re, mae_re, mape_re, r2_re
