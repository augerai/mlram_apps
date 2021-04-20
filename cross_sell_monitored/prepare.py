import helper

helper.split_to_train_test()
algorithm, score = helper.build_model('../files/cross_sell_train.csv.gz')
print(algorithm, score)
res = helper.deploy_monitored_model("cross_sell_monitored", algorithm, score)
print(res)
