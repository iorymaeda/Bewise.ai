import pandas as pd
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import RedirectResponse

from utils.processing import process_df

app = FastAPI()

class output_model(BaseModel):
    greeting: bool
    farewell: bool
    farewell_text: str
    greeting_text: str
    name_text: str
    PER_name: list[str]
    ORG_name: list[str]
    is_polite: bool


@app.get("/")
def read_root():
    return RedirectResponse(url='/docs')
    

@app.post("/process_csv/", response_model=dict[int, output_model])
async def process_csv(csv_file: UploadFile = File(...)) -> dict[int, output_model]:
    df = pd.read_csv(csv_file.file)
    result = await process_df(df)
    return result