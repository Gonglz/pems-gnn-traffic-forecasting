
import pyarrow.parquet as pq
pf = pq.ParquetFile('/scratch/lgong1/finalproject/pems_data/step51_interpolated_final.parquet')
print(pf.schema)        # 打印所有列名和类型
print(pf.schema.names)  # 只打印列名列表
