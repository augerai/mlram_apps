import helper

cur_model_path = "../models/current_model.txt"

model_id=''
with open(cur_model_path, "r") as file:
    model_id = file.read()

if model_id:
    helper.undeploy_monitored_model(model_id)

    with open(cur_model_path, "w") as file:
        file.write("")

helper.split_to_train_test()
algorithm, score, _ = helper.build_model('../files/cross_sell_train.csv.gz', 'cross_sell_xgboost_base')
print(algorithm, score)
res = helper.deploy_monitored_model("cross_sell_monitored", algorithm, score)
print(res)

model_id = res['data']['model_id']
with open(cur_model_path, "w") as file:
    file.write(model_id)
