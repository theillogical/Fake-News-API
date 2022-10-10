from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from nltk.stem import WordNetLemmatizer
from nltk import data
import joblib
import sys

data.path.append('C:\\Users\\sarth\\Downloads\\news detection\\nltk_data\\')


class Model:
    def __init__(self, model={}, tfvec={}):
        self.filename = 'TA40final.joblib'
        self.model = model
#         self.vectorization = TfidfVectorizer()

    def load(self):
        self.model = joblib.load(self.filename)

    # This is the code can be called on the server, ie. in the api route. The model should be passed to it and data from the user can be retrieved from request body or query params.
    def predict(self, data):
        return round(self.model.predict_proba([data])[0][1]*10)

    # Call this method to save the model on the disk.
    def save(self):
        joblib.dump(self.model, self.filename)


def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Perform lemmatization
    4. Returns a list of the cleaned text
    """
    lemma = WordNetLemmatizer()
    nopunc = lemma.lemmatize(mess)
    # Now just remove any stopwords
    return nopunc


model = Model()
model.load()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Data(BaseModel):
    text: str


@app.get("/")
async def root_get():
    return {"message": "Hello World"}


@app.post("/")
async def root_post(text: Data):
    print(text.text)
    prediction = model.predict(text.text)
    return {"prediction": prediction}



PORT = int(sys.argv[-1].split("=")[1]
           ) if sys.argv[-1].startswith("--port") else 8080
uvicorn.run(app, host="0.0.0.0", port=PORT, reload=True)
