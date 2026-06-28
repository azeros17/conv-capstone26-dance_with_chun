import pickle
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ori_model import OriModel
from gen_model import GenModel

app = FastAPI(title="바이오차 품질 예측")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

with open("model_ori.pkl", "rb") as f:
    ori_pipeline = pickle.load(f)

with open("model_gen.pkl", "rb") as f:
    gen_pipeline = pickle.load(f)


class Req(BaseModel):
    우분: float = 0
    돈분: float = 0
    커피박: float = 0
    톱밥: float = 0
    슬러지: float = 0
    탄화온도: float = 500
    월: int = 6


def _comp(req):
    return {"우분": req.우분, "돈분": req.돈분, "커피박": req.커피박,
            "톱밥": req.톱밥, "슬러지": req.슬러지}


@app.get("/")
def page_ori():
    return FileResponse("index_ori.html")


@app.get("/gen")
def page_gen():
    return FileResponse("index_gen.html")


@app.post("/predict")
def predict_ori(req: Req):
    return ori_pipeline.predict(_comp(req), req.탄화온도, req.월)


@app.post("/predict_gen")
def predict_gen(req: Req):
    return gen_pipeline.predict(_comp(req), req.탄화온도, req.월)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
