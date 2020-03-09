import os
import json


if __name__ == '__main__':
    with open('./results/xgboost_res.json', 'r') as f:
        scores = json.load(f)

    scores = reversed(sorted(scores, key = lambda i: i['acc']))

    for score in scores:
        print(score)
