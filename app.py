import os
from flask import Flask, jsonify, request
# from flask_restful import Api, Resource

# import sqlalchemy
import pandas as pd
import numpy as np
import math 
from sklearn import preprocessing
import pickle

app = Flask(__name__)
# api = Api(app)

@app.route("/predict", methods=['POST'])

def post():
        # Connect to Database
        # eng = sqlalchemy.create_engine('postgresql://postgres:fk051098@127.0.0.1:5432/REALESTATE')
        # a = 'select re."floor", tp."distanceFromBTS", re."buildingFloor",re."camFee", re."buildingAge", re."pricebyGov", re."materialDesign", re."units", re."areaRoom", fa."lobby", fa."lift", fa."swimmingPool", fa."fitness", fa."suana", fa."jacuzzi", fa."steam", fa."library", fa."kidplay", fa."garden", fa."parklot", fa."automateParklot", fa."golfCourse", fa."movieRoom", fa."shop",re."roomPosition", re."roomType", re."roomView",  re."buildingControlAct",  re."districtID", re."subdistrictID", tp."haveBTS", tp."haveMRT", tp."haveBRT" ,re."latitude", re."longtitude", re."buildingCondition" from public."REALESTATE" re join public."FACILITY" fa on re."projectID" = fa."projectID" join public."TRANSPORT" tp on re."projectID" = tp."projectID" WHERE re."projectID" = (select max("projectID") from public."REALESTATE")'
        # b = 'select dt."latCentreDistrict", dt."longCentreDistrict" from public."DISTRICT" dt'
        # df = pd.read_sql_query(a,eng)
        # bkk = pd.read_sql_query(b,eng)
        df = request.json['datas']
        bkk = request.json['bkk']
        print('Load Data...')
        df = pd.DataFrame.from_dict(df, orient="index")
        df = df.T
        print(df)

        bkk = pd.DataFrame.from_dict(bkk, orient="index")
        bkk = bkk.T
        print(bkk)


        df['distanceFromBTS'] =  df['distanceFromBTS'].fillna(9999)

        # lat = df['latitude'][:]
        # long = df['longtitude'][:]
        #Watthana
        lat0 = bkk['latCentreDistrict'][0]
        long0 = bkk['longCentreDistrict'][0]
        #Sathorn
        lat1 = bkk['latCentreDistrict'][1]
        long1 = bkk['longCentreDistrict'][1]
        #Dusit
        lat2 = bkk['latCentreDistrict'][2]
        long2 = bkk['longCentreDistrict'][2]
        #Bangna
        lat3 = bkk['latCentreDistrict'][3]
        long3 = bkk['longCentreDistrict'][3]
        #Phayathai
        lat5 = bkk['latCentreDistrict'][5]
        long5 = bkk['longCentreDistrict'][5]
        #Thonburi
        lat8 = bkk['latCentreDistrict'][8]
        long8 = bkk['longCentreDistrict'][8]
        #Bangrak
        lat10 = bkk['latCentreDistrict'][10]
        long10 = bkk['longCentreDistrict'][10]
        #Khlongsan
        lat15 = bkk['latCentreDistrict'][15]
        long15 = bkk['longCentreDistrict'][15]
        #Khlongtoei
        lat16 = bkk['latCentreDistrict'][16]
        long16 = bkk['longCentreDistrict'][16]
        #Pathumwan
        lat23 = bkk['latCentreDistrict'][23]
        long23 = bkk['longCentreDistrict'][23]
        #Prakhanong
        lat24 = bkk['latCentreDistrict'][24]
        long24 = bkk['longCentreDistrict'][24]
        #Ratchatewee
        lat27 = bkk['latCentreDistrict'][27]
        long27 = bkk['longCentreDistrict'][27]
        d0 = [] 
        d1 = [] 
        d2 = []
        d3 = []
        d5 = []
        d8 = []
        d10 = []
        d15 = []    
        d16 = []
        d23 = []
        d24 = []
        d27 = [] 
        d = []

        for i ,j in zip(df['latitude'] , df['longtitude']):
            d0.append(math.sqrt(((lat0-i)**2)+((long0-j)**2)))
            d1.append(math.sqrt(((lat1-i)**2)+((long1-j)**2)))
            d2.append(math.sqrt(((lat2-i)**2)+((long2-j)**2)))
            d3.append(math.sqrt(((lat3-i)**2)+((long3-j)**2)))
            d5.append(math.sqrt(((lat5-i)**2)+((long5-j)**2)))
            d8.append(math.sqrt(((lat8-i)**2)+((long8-j)**2)))
            d10.append(math.sqrt(((lat10-i)**2)+((long10-j)**2)))
            d15.append(math.sqrt(((lat15-i)**2)+((long15-j)**2)))
            d16.append(math.sqrt(((lat16-i)**2)+((long16-j)**2)))
            d23.append(math.sqrt(((lat23-i)**2)+((long23-j)**2)))
            d24.append(math.sqrt(((lat24-i)**2)+((long24-j)**2)))
            d27.append(math.sqrt(((lat27-i)**2)+((long27-j)**2)))
    
            d.append(np.array([d0,d1,d2,d3,d5,d8,d10,d15,d16,d23,d24,d27]))
            float()
        df = df.assign(d0 = d0)
        df = df.assign(d1 = d1)
        df = df.assign(d2 = d2)
        df = df.assign(d3 = d3)
        df = df.assign(d5 = d5)
        df = df.assign(d8 = d8)
        df = df.assign(d10 = d10)
        df = df.assign(d15 = d15)
        df = df.assign(d16 = d16)
        df = df.assign(d23 = d23)
        df = df.assign(d24 = d24)
        df = df.assign(d27 = d27)

        X =  df.values
        X = pd.DataFrame(preprocessing.normalize(X))

        results = model.predict(X)

        print ("results is ", results)

        return jsonify({ "result" : results})


if __name__ == "__main__":
    with open('baggingclassificaion_weight.pkl', 'rb') as f:
        model = pickle.load(f)
    print("download model success")
    print("...")
    post()
    app.run(host='0.0.0.0', port=5000, debug=True)