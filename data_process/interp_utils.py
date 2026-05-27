import pandas as pd

def global_fill(df: pd.DataFrame,
                features: list,
                mask_flag_col: str = 'mask_flag') -> pd.DataFrame:
    """
    note(note station_id+direction)noterows"note":
    - mask_flag == False noterowsnote"noterows", notecomputenote;
    - mask_flag == True noterows, noterows, noterowsnote;
      note *note* noterows, note, note(note NaN).
    """
    valid = df.loc[~df[mask_flag_col], features]
    if valid.shape[0] == 0:
        # note mask ⇒ note
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
    noterows, note mask_flag==True, note nbr_idx noterowsnote
    mask_flag==False note feature note NaN note feature.
    note, note(note NaN).
    note DataFrame, note df.
    """
    out = df.copy()
    for i, row in df.iterrows():
        if row[mask_flag_col]:
            neigh = row.get(nbr_col) or []
            # note -> note feature note
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
    note group_cols note, note time_col note, note feature note mask_flag==True note; note mask note, note.
    note DataFrame, note df.
    """
    out = df.copy()
    # note timestamp note
    ts_num = pd.to_datetime(out[time_col]).astype(np.int64)
    out['_ts_num_'] = ts_num

    # noteresultnote
    filled = pd.Series(index=out.index, dtype=float)

    # note
    for _, g in out.groupby(group_cols):
        idx = g.index
        t = g['_ts_num_'].values
        v = g[feature].values.astype(float)
        m = g[mask_flag_col].values.astype(bool)

        # note, note
        valid = (~m) & ~np.isnan(v)
        if valid.sum() <= 1:
            # note
            filled.loc[idx] = v
        else:
            # note: note m==True note, note t,v note valid note
            vi = v.copy()
            xi = t[m]
            vi[m] = np.interp(xi, t[valid], v[valid])
            filled.loc[idx] = vi

    # notecleanup
    out[feature] = filled
    out.drop(columns=['_ts_num_'], inplace=True)
    return out
