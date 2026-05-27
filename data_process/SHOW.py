
import pyarrow.parquet as pq
pf = pq.ParquetFile('/scratch/lgong1/finalproject/pems_data/step51_interpolated_final.parquet')
print(pf.schema)        # noteclassnote
print(pf.schema.names)  # note
