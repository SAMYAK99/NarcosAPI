from flask import Flask,request,jsonify
import numpy as np
import pickle

model = pickle.load(open('model.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello world"

@app.route('/predict',methods=['POST'])
def predict():
    age = int(request.form.get('age'))
    gender = int(request.form.get('gender'))
    education = int(request.form.get('education'))
    enjoyableWith = int(request.form.get('enjoyableWith'))
    liveWith = int(request.form.get('liveWith'))
    spendMostTime = int(request.form.get('spendMostTime'))
    failueInLife = int(request.form.get('failueInLife'))
    sucidalThoughts = int(request.form.get('sucidalThoughts'))
    familyRelationship = int(request.form.get('familyRelationship'))
    addictedPersonInFamily = int(request.form.get('addictedPersonInFamily'))
    noOfFriends = int(request.form.get('noOfFriends'))
    satisfiedWithWorkplace = int(request.form.get('satisfiedWithWorkplace'))
    caseInCourt = int(request.form.get('caseInCourt'))
    livingWithDrugUser = int(request.form.get('livingWithDrugUser'))
    smoking = int(request.form.get('smoking'))
    everTakenDrug = int(request.form.get('everTakenDrug'))
    friendInfluence = int(request.form.get('friendInfluence'))
    chanceGivenToTaste = int(request.form.get('chanceGivenToTaste'))
    easyToControl = int(request.form.get('easyToControl'))

    input_query = np.array([[age,gender,education,enjoyableWith,liveWith,spendMostTime,failueInLife,sucidalThoughts,familyRelationship,addictedPersonInFamily,noOfFriends,satisfiedWithWorkplace,caseInCourt,livingWithDrugUser,smoking,everTakenDrug,friendInfluence,chanceGivenToTaste,easyToControl]])

    result = model.predict(input_query)[0]

    return jsonify({'addicted':str(result)})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
