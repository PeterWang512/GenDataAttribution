import os
import glob
import pickle
import pandas as pd


def get_dataframe(data):
    columns = ['name', 'feature']
    data_keys = list(data.keys())
    recall_ks = data[data_keys[0]]['recall_ks']
    for k in data.keys():
        for sz in recall_ks:
            columns.append(f'{k}-recall@{sz}')
        columns.append(f'{k}-mAP')

    return pd.DataFrame(columns=columns), recall_ks


categories = ['artchive', 'bamfg', 'observed_imagenet', 'unobserved_imagenet']
feats = ['clip', 'dino', 'moco', 'vit', 'sscd']
for c in categories:
    subset = sorted(glob.glob(f'results/test_{c}/*.pkl'))

    df, recall_ks = None, []
    for ff in feats:
        small_subset = sorted([n for n in subset if ff in n])
        # print(subset)
        for s in small_subset:
            print(s)
            entry = {}
            with open(s, 'rb') as f:
                data = pickle.load(f)

            # lazy load the dataframe
            if df is None or recall_ks == []:
                df, recall_ks = get_dataframe(data)
            
            name = os.path.basename(s).replace('.pkl', '')
            entry['name'] = name
            entry['feature'] = ff
            for k in data.keys():
                for ind, tt in enumerate(recall_ks):
                    entry[f'{k}-recall@{tt}'] = data[k]['avg_recall'][ind]
                entry[f'{k}-mAP'] = data[k]['avg_map']
            
            df = df.append(entry, ignore_index=True)
    
    df.to_csv(f'results/table_{c}.csv')
