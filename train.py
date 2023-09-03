import json

import model
import pandas as pd

def train_and_save_model(fn, verbose=True):
    imdb_path = './data/imdb/'
    json_list = []
    total_cost = []
    #for i in range(20): 2
    file = imdb_path + 'plan_and_cost/train_plan_part{}.csv'.format(3)
    df = pd.read_csv(file)
    json_column = df['json'].values.tolist()
    for plan_json in json_column:
        json_list.append(plan_json)
        parse_json = json.loads(plan_json)
        total_cost.append(parse_json['Plan']['Total Cost'])
    print("***********************************************")
    reg = model.BaoRegression(have_cache_data=False, verbose=verbose)
    reg.fit(json_list, total_cost)
    reg.save(fn)
    return reg



if __name__ == "__main__":
    # json_value = {"Plan": {"Node Type": "Gather", "Parallel Aware": False, "Startup Cost": 23540.58, "Total Cost": 154548.95, "Plan Rows": 567655, "Plan Width": 119, "Actual Startup Time": 386.847, "Actual Total Time": 646.972, "Actual Rows": 283812, "Actual Loops": 1, "Workers Planned": 2, "Workers Launched": 2, "Single Copy": False, "Plans": [{"Node Type": "Hash Join", "Parent Relationship": "Outer", "Parallel Aware": True, "Join Type": "Inner", "Startup Cost": 22540.58, "Total Cost": 96783.45, "Plan Rows": 236523, "Plan Width": 119, "Actual Startup Time": 369.985, "Actual Total Time": 518.487, "Actual Rows": 94604, "Actual Loops": 3, "Inner Unique": False, "Hash Cond": "(t.id = mi_idx.movie_id)", "Workers": [], "Plans": [{"Node Type": "Seq Scan", "Parent Relationship": "Outer", "Parallel Aware": True, "Relation Name": "title", "Alias": "t", "Startup Cost": 0.0, "Total Cost": 49166.46, "Plan Rows": 649574, "Plan Width": 94, "Actual Startup Time": 0.366, "Actual Total Time": 147.047, "Actual Rows": 514421, "Actual Loops": 3, "Filter": "(kind_id = 7)", "Rows Removed by Filter": 328349, "Workers": []}, {"Node Type": "Hash", "Parent Relationship": "Inner", "Parallel Aware": True, "Startup Cost": 15122.68, "Total Cost": 15122.68, "Plan Rows": 383592, "Plan Width": 25, "Actual Startup Time": 103.547, "Actual Total Time": 103.547, "Actual Rows": 306703, "Actual Loops": 3, "Hash Buckets": 65536, "Original Hash Buckets": 65536, "Hash Batches": 32, "Original Hash Batches": 32, "Peak Memory Usage": 1920, "Workers": [], "Plans": [{"Node Type": "Seq Scan", "Parent Relationship": "Outer", "Parallel Aware": True, "Relation Name": "movie_info_idx", "Alias": "mi_idx", "Startup Cost": 0.0, "Total Cost": 15122.68, "Plan Rows": 383592, "Plan Width": 25, "Actual Startup Time": 0.28, "Actual Total Time": 54.382, "Actual Rows": 306703, "Actual Loops": 3, "Filter": "(info_type_id > 99)", "Rows Removed by Filter": 153308, "Workers": []}]}]}]}, "Planning Time": 2.382, "Triggers": [], "Execution Time": 654.241}
    # print(json_value['Plan']['Total Cost'])
    #train_and_save_model("bao_model")
    model = model.BaoRegression()
    model.load("bao_model")
    imdb_path = "./data/imdb/"
    file = imdb_path + 'plan_and_cost/train_plan_part{}.csv'.format(0)
    df = pd.read_csv(file)
    json_list = df['json'].values.tolist()
    result = model.predict(json_list)
    print(result)


