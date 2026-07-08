from fastapi import FastAPI
import joblib 
import pandas as pd

app = FastAPI()

model = joblib.load("titanic_prediction_model.pkl")

# data = pd.DataFrame({
#     "Age":[23], 
#     "Pclass":[1], 
#     "SibSp":[0],
#     "Parch":[1],
#     "Gender":[0]

# })
# output = model.predict(data)

# print(output)


@app.get("/")
def index():
    return {"Hello": "World"}


@app.get("/predict")
def predict(age,gender,p_class,sib_sp,parent_child):
    
    data = pd.DataFrame({
        "Age":[age], 
        "Pclass":[p_class], 
        "SibSp":[sib_sp],
        "Parch":[parent_child],
        "Gender":[0 if gender == "Male" else 1]
    })

    print("data"+str(data))
    output = model.predict(data)
    
    return {"prediction":int(output[0])}

