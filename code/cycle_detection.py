import pandas as pd
from matplotlib import pylab as pl
import matplotlib.pyplot as plt
import networkx as nx
from networkx import *
from datetime import datetime
import itertools
from tqdm import tqdm
import json
import requests
from itertools import product
from collections import Counter
from scipy import stats
from networkx.algorithms.components.connected import connected_components

def read_dataset(tx_all):
    tx_all = pd.read_csv(file)
    tx_all = tx_all.sort_values(by = ['block', 'txn_hash_idx'])
    tx_all = tx_all[['block', 'from_addr', 'to_addr', 'token_id',
           'tx_hash', 'value', 'timestamp', 'exchange', 'eth_value', 'usd_value']]

    tx_all['exchange'] = tx_all['exchange'].apply(lambda x: x if str(x)!='nan' else 'NA')
    tx_all['day'] = tx_all['timestamp'].apply(lambda x: datetime.fromtimestamp(x))
    tx_all['date'] = tx_all['day'].apply(lambda x: x.date())

    # assign tx a number
    tx_all['tx_index'] = tx_all.index
    
    # assign node number
    all_addr = list(set(list(tx_all.from_addr) + list(tx_all.to_addr)))
    all_addr.sort()
    w = 1
    addr_to_id = {}
    id_to_addr = {}
    for i in all_addr:
        addr_to_id[i] = w
        id_to_addr[w] = i
        w += 1

    tx_all['from_id'] = tx_all['from_addr'].apply(lambda x: addr_to_id[x])
    tx_all['to_id'] = tx_all['to_addr'].apply(lambda x: addr_to_id[x])
    
    return tx_all, addr_to_id

def detect_token_path(tx_all):
    tokens = list(set(tx_all.token_id.values))

    token_path = {}

    for t in tokens:
        t_data = tx_all.query('token_id == @t')

        start = t_data.iloc[0]
        tx_path = [(start.from_id, start.tx_index)]

        for i in range(t_data.shape[0]):
            data = t_data.iloc[i]
            tx_path.append((data.to_id, data.tx_index))

        token_path[t] = tx_path
    
    return token_path

def within_k_days(start, end, k, tx_all):
    """Test whether a cycle is completed within one day """
    Txindex_to_Timestamp = dict(zip(tx_all.tx_index, tx_all.timestamp))
    ts1, ts2 = Txindex_to_Timestamp[start[1]], Txindex_to_Timestamp[end[1]]
    
    return (ts2 - ts1) < 86400 * k

def find_cycles(lst, tx_all):
    visited = {}
    path = []
    cycles = []
    for i in range(len(lst)):
        node = lst[i]
        id_ = node[0]
        path.append(node)
        
        if id_ in visited:
            last_index = visited[id_][-1]
            new_cycles = path[last_index+1: i+1] 
            if len(new_cycles) == 1:
                continue
            
            if within_k_days(new_cycles[0], new_cycles[-1], 30, tx_all):
                new_cycles_index = [n[1] for n in new_cycles]
                cycles.append(new_cycles_index)
            
            visited[id_].append(i) 
            
        else:
            visited[id_] = [i]
            
    return cycles      

def detect_cycles(tx_all, contract):
    token_path = detect_token_path(tx_all)
    tokens = list(set(tx_all.token_id.values))
    
    token_cycles = {}
    for t in tokens:
        path = token_path[t]
        cycles = find_cycles(path, tx_all)
        token_cycles[t] = cycles   
    all_cycles = list(itertools.chain.from_iterable(list(token_cycles.values())))
    
    cycles_clean = []
    contract_tx_id = list(tx_all.query('from_addr in @contract or to_addr in @contract').tx_index.values)
    
    for cycle in all_cycles:
        con = [k for k in cycle if k in contract_tx_id]
        if len(con) == 0: cycles_clean.append(cycle)

    return cycles_clean 

def calulate_statistic(tx_all, cycles_clean_sale, contract, nft):
    cycles_clean_sale = list(itertools.chain.from_iterable(cycles_clean_sale))
    cycles_clean_sale = list(set(cycles_clean_sale))
    tx_wt = tx_all.query('tx_index in @cycles_clean_sale')
    tx_wt.to_csv('../results/wt/wash_trade_txs_{}.csv'.format(nft), index = 0)
    
    wt_addr = set(list(tx_wt.from_addr) + list(tx_wt.to_addr))
    all_addr = list(set(list(tx_all.from_addr) + list(tx_all.to_addr)))
    exclude = contract.copy()
    all_addr_clean = set([i for i in all_addr if i not in exclude])
    abn_addr = len(wt_addr)/len(all_addr_clean)

    wt_tokens = set(list(tx_wt.token_id))
    all_tokens = set(list(tx_all.token_id))
    abn_tokens = len(wt_tokens)/len(all_tokens)

    tx_all = tx_all.drop_duplicates(subset = ['tx_hash'])
    tx_wt = tx_wt.drop_duplicates(subset = ['tx_hash'])
    abn_vol = tx_wt.usd_value.sum()/tx_all.usd_value.sum()
    abn_sales_num = tx_wt.query('usd_value != 0').shape[0]/tx_all.query('usd_value != 0').shape[0]
    abn_tx_num = tx_wt.shape[0]/tx_all.shape[0]
    abn_usd = tx_wt.usd_value.sum()
   
    return [nft, abn_usd,
            len(wt_addr), len(all_addr_clean), abn_addr, 
            tx_wt.usd_value.sum(), tx_all.usd_value.sum(), abn_vol,
            tx_wt.query('usd_value != 0').shape[0], tx_all.query('usd_value != 0').shape[0], abn_sales_num, 
            tx_wt.shape[0], tx_all.shape[0], abn_tx_num, 
            len(wt_tokens), len(all_tokens), abn_tokens]


nft_address_path = '../data/nft.txt'
with open(nft_address_path) as f:
    nfts = f.readlines()
nfts = [i.replace('\n', '') for i in nfts]

contract_path = '../data/contract.txt'
with open(contract_path) as f:
    contract = f.readlines()
contract = [i.replace('\n', '') for i in contract]
contract = list(set(contract))
contract.append('0x0000000000000000000000000000000000000000')

fw = open('../results/summary.csv', 'w')

for nft in tqdm(nfts):
    # print(nft)
    file = '../data/combine/{}.csv'.format(nft)
    tx_all, addr_to_id = read_dataset(file)
    cycles = detect_cycles(tx_all, contract)
    res = calulate_statistic(tx_all, cycles, contract, nft)
    lines = ','.join(str(e) for e in res)+'\n'
    fw.writelines(lines)
    
