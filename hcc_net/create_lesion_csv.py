import pandas as pd
from config import CFG

phase_df = pd.read_csv(CFG.PHASE_CSV)
patient_df = pd.read_csv(CFG.CSV_PATH)

prio = ["P", "C3", "C2", "C1"]
phase_rank = {p:i for i,p in enumerate(prio)}

lesion_mask_choices = (
    phase_df.sort_values(by=["patient_id"], kind="stable")
            .assign(phase_order=phase_df["phase"].map(phase_rank).fillna(99))
            .sort_values(["patient_id","phase_order"])
            .groupby("patient_id")["mask_path"]
            .agg(lambda x: next((v for v in x if isinstance(v,str) and len(v)>0), ""))
            .reset_index()
)

out = patient_df.merge(lesion_mask_choices, on="patient_id", how="left")

OUT_PATH = r"D:\HCC\patient_rows_with_lesion.csv"
out.to_csv(OUT_PATH, index=False)

print("Saved:", OUT_PATH)
print("Rows:", len(out))
print(out.head())

