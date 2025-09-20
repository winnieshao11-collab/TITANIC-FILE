from flask import Flask, render_template,request
import joblib
 
app =Flask(__name__)
model = joblib.load("titanic_model.pkl")
 
@app.route("/")
def home():
    return render_template("index.html")
 
@app.route("/predict",methods = ["POST"])
def predict():
    sex = 1 if request.form["sex"]=="female" else 0
    fare = float(request.form["fare"])
    prediction = model.predict([[sex,fare]])[0]
    result = "Survived" if prediction == 1 else "Did not Survive"
    return render_template("index.html",prediction = result)
 
if __name__ == "_main_":
    app.run(debug = True)
    