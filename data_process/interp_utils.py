import pandas as pd

def global_fill(df: pd.DataFrame,
                features: list,
                mask_flag_col: str = 'mask_flag') -> pd.DataFrame:
    """
    对单个组（即同一个 station_id+direction）进行“组内平均填充”：
    - mask_flag == False 的行被视作“有效行”，它们用来计算均值；
    - mask_flag == True 的行，如果有有效行，则填充为有效行的平均值；
      如果 *没有* 有效行，则什么都不做，保留原值（可能是 NaN）。
    """
    valid = df.loc[~df[mask_flag_col], features]
    if valid.shape[0] == 0:
        # 全部都是 mask ⇒ 不做任何改动
        return df

    means = valid.mean()
    mask_idx = df[mask_flag_col]
    df.loc[mask_idx, features] = df.loc[mask_idx, features].fillna(means)
    return df
import pandas as pd
import numpy as np

def local_fill(df: pd.DataFrame,
               feature: str,
               nbr_col: str = 'nbr_idx',
               mask_flag_col: str = 'mask_flag') -> pd.DataFrame:
    """
    对每一行，如果 mask_flag==True，就取其 nbr_idx 列指定的邻居行中
    mask_flag==False 且该 feature 非 NaN 的值的平均值来填充 feature。
    如果找不到任何有效邻居，则保留原值（可能是 NaN）。
    返回新的 DataFrame，不修改原 df。
    """
    out = df.copy()
    for i, row in df.iterrows():
        if row[mask_flag_col]:
            neigh = row.get(nbr_col) or []
            # 索引列表 → 找出所有有效邻居的 feature 值
            vals = [
                df.at[j, feature]
                for j in neigh
                if (j in df.index)
                   and not df.at[j, mask_flag_col]
                   and pd.notna(df.at[j, feature])
            ]
            if vals:
                out.at[i, feature] = float(np.mean(vals))
    return out
import pandas as pd
import numpy as np
from typing import List

def temporal_fill(
    df: pd.DataFrame,
    feature: str,
    group_cols: List[str] = ['station_id', 'direction'],
    time_col: str = 'timestamp',
    mask_flag_col: str = 'mask_flag'
) -> pd.DataFrame:
    """
    对每个 group_cols 分组内，按 time_col 排序，对 feature 列中 mask_flag==True 的位置
    做线性时间插值；若某组全为 mask 或仅有一条有效，则保留原值。
    返回一个新的 DataFrame，不修改原 df。
    """
    out = df.copy()
    # 确保 timestamp 可做数值插值
    ts_num = pd.to_datetime(out[time_col]).astype(np.int64)
    out['_ts_num_'] = ts_num

    # 准备结果容器
    filled = pd.Series(index=out.index, dtype=float)

    # 对每个分组做插值
    for _, g in out.groupby(group_cols):
        idx = g.index
        t = g['_ts_num_'].values
        v = g[feature].values.astype(float)
        m = g[mask_flag_col].values.astype(bool)

        # 如果组内没有有效点或仅一个有效点，跳过插值
        valid = (~m) & ~np.isnan(v)
        if valid.sum() <= 1:
            # 直接保留原值
            filled.loc[idx] = v
        else:
            # 插值：对 m==True 的位置，用 t,v 中 valid 部分插值得到
            vi = v.copy()
            xi = t[m]
            vi[m] = np.interp(xi, t[valid], v[valid])
            filled.loc[idx] = vi

    # 写回并清理
    out[feature] = filled
    out.drop(columns=['_ts_num_'], inplace=True)
    return out
