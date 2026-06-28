import pickle
import os
from ori_model import OriModel

CSV_DIR = "."

B_CSV = os.path.join(CSV_DIR, "3_탄화온도_B.csv")
C_CSV = os.path.join(CSV_DIR, "4_가스연속_C.csv")
SUPP_CSV = os.path.join(CSV_DIR, "9_원료배합별_바이오차_비료.csv")

pipe = OriModel().train(B_CSV, C_CSV, SUPP_CSV, verbose=False)

with open("model_ori.pkl", "wb") as f:
    pickle.dump(pipe, f)

print("model_ori.pkl 저장 완료")
