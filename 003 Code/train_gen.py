import pickle
import os
from gen_model import GenModel

CSV_DIR = "."

PROP_CSV = os.path.join(CSV_DIR, "데이터_물성_조성온도.csv")
GAS_CSV = os.path.join(CSV_DIR, "4_가스연속_C.csv")

pipe = GenModel().train(PROP_CSV, GAS_CSV, verbose=False)

with open("model_gen.pkl", "wb") as f:
    pickle.dump(pipe, f)

print("model_gen.pkl 저장 완료")
