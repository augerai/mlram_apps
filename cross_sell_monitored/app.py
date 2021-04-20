import pandas as pd
from datetime import datetime, timedelta
import time
import joblib

from xgboost import XGBClassifier

from a2ml.api.a2ml import A2ML, Context
from helper import build_model, deploy_monitored_model

days = 20 #38
day_size = 9500

target = "Response"

df = pd.read_csv('../files/cross_sell_test.csv.gz')
print(len(df))

model_id='3589896dd31c2d99'
model_name='cross_sell_monitored'
ctx = Context()
a2ml = A2ML(ctx)

a2ml.delete_actuals(model_id)

start_date = datetime.strptime("2020-12-01","%Y-%m-%d").date()

model = joblib.load('../models/cross_sell_xgboost.pkl')

for day in range(0,days):
    pos = day*day_size
    data = df[pos:pos+day_size]
    actuals = data[target]
    del data[target]

    predictions = model.predict(data)

    data[target] = predictions
    data['actual'] = actuals

    a2ml.actuals(model_id, data=data.to_dict('list'), actuals_at = str(start_date))
    start_date = start_date+timedelta(days=1)

    print("Wait 30 sec to update review status.")
    time.sleep(30)
    result = a2ml.review(model_id)
    print("Review result: %s"%result)
    status = result['data'].get('status')

    if not status:
        print("No retrain for day: %d"%day)
        continue

    if result['data'].get('status') == 'retrain':
        print("Do local model retrain with new data and deploy new model.")
        algorithm, score = build_model('../files/cross_sell_train.csv.gz', df[pos:pos+day_size])
        result = deploy_monitored_model("cross_sell_monitored", algorithm, score)
        model_id = result['data']['model_id']
        print("New model: %s, algorithm: %s, score: %s"%(model_id, algorithm, score))
