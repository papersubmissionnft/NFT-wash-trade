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

def read_dataset(file):
    tx_all = pd.read_csv(file)
    tx_all = tx_all.sort_values(by = ['block', 'txn_hash_idx'])
    tx_all = tx_all[['block', 'from_addr', 'to_addr', 'token_id',
           'tx_hash', 'value', 'timestamp', 'usd_value', 'exchange', 'eth_value']]
    
    tx_all['exchange'] = tx_all['exchange'].apply(lambda x: x if str(x)!='nan' else 'NA')
    tx_all['day'] = tx_all['timestamp'].apply(lambda x: datetime.fromtimestamp(x))
    tx_all['date'] = tx_all['day'].apply(lambda x: x.date())

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

def create_graph_based_on_path(path, tx_all, t): 
    """Create networkx graph using the transaction path"""

    tx_hash = [i[2] for i in path]
    tx = tx_all.query('tx_hash in @tx_hash and token_id == @t')
    g = nx.from_pandas_edgelist(tx, source='from_id', target='to_id', 
                             create_using=nx.DiGraph, 
                             edge_attr=['token_id', 'tx_hash', 'date'])
    return g

def find_cycle_prices(lst, tx_all, token):
    index = [i[2] for i in lst]
    tx = tx_all.query('tx_hash in @index and token_id == @token')
    return (index, tx.usd_value.sum()) 

def detect_token_path(tx_all):
    tokens = list(set(tx_all.token_id.values))
    token_path = {}
    for t in tokens:
        t_data = tx_all.query('token_id == @t')
        tx_path = []
        for i in range(t_data.shape[0]):
            data = t_data.iloc[i]
            pair = (data.from_id, data.to_id, data.tx_hash)
            tx_path.append(pair)
        token_path[t] = tx_path
    return token_path

def within_k_days(start, end, k, tx_all):
    """Test whether a cycle is completed within one day"""
    Txhash_to_Timestamp = dict(zip(tx_all.tx_hash, tx_all.timestamp))
    ts1, ts2 = Txhash_to_Timestamp[start], Txhash_to_Timestamp[end]
    return (ts2 - ts1) < 86400 * k

def find_cycles(path, k, tx_all):
    visited = {path[1][0]: 1} 
    cycles = []
    
    i = 1
    while i < len(path):
        from_, to_, tx_hash = path[i][0], path[i][1], path[i][2]
        
        if to_ in visited:
            index = visited[to_]
            new_cycles = path[index: i+1] 
            
            if within_k_days(new_cycles[0][2], new_cycles[-1][2], 30, tx_all):
                cycles.append(new_cycles)
                visited = {} 
        else:
            visited[from_] = i
        
        i += 1
            
    return cycles   

def combine_cycle(last_g, new, tx_all, token): 
    new_g = create_graph_based_on_path(new, tx_all, token)
    
    if len(last_g) == 0:
        return (new, [new_g], False)

    last, last_patterns = last_g[0], last_g[1]

    last_n = set([i[0] for i in last] + [i[1] for i in last]) 
    new_n = set([i[0] for i in new] + [i[1] for i in new]) 
    common = [i for i in new_n if i in last_n] 

    if len(common) == 0:
        return (new, [new_g], False)
    
    elif within_k_days(last[0][-1], new[-1][-1], 30, tx_all) == False:
        return (new, [new_g], False)

    else:
        for p in last_patterns:
            if nx.is_isomorphic(p, new_g) and list(p.nodes()) == list(new_g.nodes()):
                return (new, [new_g], False)          
        last.extend(new)
        last_patterns.append(new_g)

    return (last, last_patterns, True)

def get_combined_cycles(cycles, contract_tx_hash, duplicate_tx_hash, token):
    combined = [] 
    last, update = (), ()

    i = 0
    while i < len(cycles):
        new = cycles[i]
        i += 1
        
        index = [k[2] for k in new]
        if len([m for m in index if (m in contract_tx_hash or m in duplicate_tx_hash)]) != 0:
            if last != ():
                combined.append(last[0])
            last = ()
            continue

        update = combine_cycle(last, new, tx_all, token)

        if update[2] == False: # if not combined
            # add last to result 
            if last != ():
                combined.append(last[0])
            # update last to current pattern
            last = update[0:2]

        else: # if combined
            last = update[0:2]

    if len(last) != 0:
        combined.append(last[0])
    
    return combined

def detect_all_cycles(tx_all, contract, duplicate_tx_hash, addr_to_id):
    """Return the combined cycles in [(from_id, to_id, tx_hash)...] """
    
    # get transaction path of each token
    token_path = detect_token_path(tx_all)
    
    tokens = list(set(tx_all.token_id.values))
    
    # get cycle of each token
    token_cycles = {}
    for t in tokens:
        path = token_path[t]
        cycles = find_cycles(path, 30, tx_all)
        token_cycles[t] = cycles
    
    # combine and delete cycles with contract
    contract_tx_hash = list(tx_all.query('from_addr in @contract or to_addr in @contract').tx_hash.values)
    token_combined_cycles = {}
    for t in token_cycles:
        cycles = token_cycles[t]
        combined_cycles = get_combined_cycles(cycles, contract_tx_hash, duplicate_tx_hash, t)
        token_combined_cycles[t] = combined_cycles

    # get money-flow cycle
    token_cycles_sale = {}
    for token in token_combined_cycles:
        cycles = token_combined_cycles[token]
        cycles_sale = []
        for c in cycles:
            res = find_cycle_prices(c, tx_all, token)
            if res[1] == 0:
                continue
            else:
                cycles_sale.append(c)
        token_cycles_sale[token] = cycles_sale
        
    return token_combined_cycles, token_cycles_sale

def classify_cycles(tx_all, token_cycles_clean, patterns, graph_to_txhash, nft):
    
    # create a list of cycles
    combined_cycles = []
    for token in token_cycles_clean:
        for cycles in token_cycles_clean[token]:
            c = ([i[2] for i in cycles], token)
            combined_cycles.append(c)
    
    # transform cycles to networkx graph
    g_combine = [] 
    for com in combined_cycles:
        tx_hash, token_id = com[0], com[1]
        test_tx = tx_all.query('tx_hash in @tx_hash and token_id == @token_id')
        test_g = nx.from_pandas_edgelist(test_tx, source='from_id', target='to_id', 
                                         create_using=nx.DiGraph, 
                                         edge_attr=['token_id', 'tx_hash', 'date', 'usd_value']) 
        g_combine.append((test_g, tx_hash, token_id))
        graph_to_txhash[test_g] = (tx_hash, token_id)
    
    def new_pattern(patterns, new_g):
        for g in patterns.keys():
            if nx.is_isomorphic(g, new_g):
                return g
        return 'new'

    # classify cycles
    for t in g_combine:
        this_g, tx_hash, token_id = t[0], t[1], t[2]

        if new_pattern(patterns, this_g) == 'new':
            patterns[this_g] = [(this_g, tx_hash, token_id, nft)]

        else:
            key = new_pattern(patterns, this_g)
            patterns[key].append((this_g, tx_hash, token_id, nft))
    
    return patterns, graph_to_txhash

def pattern_analysis(patterns, graph_to_txhash, tx):
    patterns_sorted = {k: v for k, v in sorted(patterns.items(), 
                                               key=lambda item: len(item[1]), reverse = True)}
        
    pattern_list = list(patterns_sorted.keys())
    Patterns_to_Index = {}
    i = 0
    for p in patterns_sorted:
        Patterns_to_Index[p] = i
        i += 1
    Index_to_Patterns = {v: k for k, v in Patterns_to_Index.items()}
    
    # create a pattern dictionary 
    pattern_txhash = {}
    for p in patterns:
        g = patterns[p]
        res = []
        for i in g:
            temp_g, tx_hash, token_id, nft = i
            res.append((tx_hash, token_id, nft))
        pattern_txhash[p] = res
    
    # transform to a dataframe
    Tx_hash_to_Price = dict(zip(tx.tx_hash, tx.usd_value))
    Tx_hash_to_ethPrice = dict(zip(tx.tx_hash, tx.eth_value))
    Tx_hash_to_Timestamp = dict(zip(tx.tx_hash, tx.timestamp))
    Tx_hash_to_Exchange = dict(zip(tx.tx_hash, tx.exchange))
    
    pattern_df, tx_hash_df, prices_df, ts_df, token_df, nft_df, exch_df, eth_df = [], [], [], [], [], [], [], []
    
    for p in pattern_txhash:
        p_index = [Patterns_to_Index[p]] * len(patterns_sorted[p])
        pattern_df.extend(p_index)

        tx_hash = [i[0] for i in pattern_txhash[p]]
        tx_hash_df.extend(tx_hash)
        
        token = [i[1] for i in pattern_txhash[p]]
        token_df.extend(token)
        
        nft = [i[2] for i in pattern_txhash[p]]
        nft_df.extend(nft)

        price, timestamp, exch, eths = [], [], [], []

        for group in tx_hash:
            pri = [Tx_hash_to_Price[i] for i in group]
            eth = [Tx_hash_to_ethPrice[i] for i in group]
            ts = [Tx_hash_to_Timestamp[i] for i in group]
            exchanges = [Tx_hash_to_Exchange[i] for i in group]

            price.append(pri)
            eths.append(eth)
            timestamp.append(ts)
            exch.append(exchanges)

        prices_df.extend(price)
        eth_df.extend(eths)
        ts_df.extend(timestamp)
        exch_df.extend(exch)
        
    wt_summary = pd.DataFrame({'pattern_index': pattern_df, 'tx_hash': tx_hash_df, 'nft': nft_df,
                               'token': token_df, 'prices': prices_df, 'eth': eth_df,  'timestamps': ts_df, 
                              'exchange': exch_df})
    
    # get next price
    def get_next_tx(row):
        token = row.token
        token_path = nft_paths[row.nft]
        path = [i[2] for i in token_path[token]]
        last_index = path.index(row.tx_hash[-1])
        next_index = path[last_index+1:]
        return next_index

    wt_summary['next_tx'] = wt_summary.apply(lambda row:get_next_tx(row), axis = 1)
    wt_summary['next_price'] = wt_summary['next_tx'].apply(lambda x: [Tx_hash_to_Price[i] for i in x])
    wt_summary['next_eth_price'] = wt_summary['next_tx'].apply(lambda x: [Tx_hash_to_ethPrice[i] for i in x])

    return wt_summary, patterns_sorted, Patterns_to_Index


patterns_all, graph_to_txhash_all = {}, {}

nft_paths = {}

nft_address_path = '../data/nft.txt'
with open(nft_address_path) as f:
    nfts = f.readlines()
nfts = [i.replace('\n', '') for i in nfts]

# read in all contract
contract_path = '../data/contract.txt'
with open(contract_path) as f:
    contract = f.readlines()
contract = [i.replace('\n', '') for i in contract]
contract.append('0x0000000000000000000000000000000000000000')

all_txs = pd.DataFrame(columns = ['block', 'from_addr', 'to_addr', 'token_id', 'tx_hash', 'value', 'timestamp', 
    'usd_value', 'exchange', 'eth_value', 'day', 'date', 'from_id', 'to_id'])

for nft in tqdm(nfts):
    file = '../data/combine/{}.csv'.format(nft)
    
    tx_all, addr_to_id = read_dataset(file)
    
    # update all_txs
    nft_paths[nft] = detect_token_path(tx_all)
    tx_all['nft'] = nft
    all_txs = all_txs.append(tx_all)
    
    # get all duplicate
    duplicate = tx_all.tx_hash.value_counts().reset_index().query('tx_hash > 1')
    duplicate.columns = ['tx_hash', 'number_of_tx']
    duplicate_tx_hash = list(duplicate.tx_hash.values)
    
    token_cycles_all,  token_cycles_clean = detect_all_cycles(tx_all, contract, duplicate_tx_hash, addr_to_id)
    patterns_all, graph_to_txhash_all = classify_cycles(tx_all, token_cycles_all, patterns_all, graph_to_txhash_all, nft)
    
wt_summary_all, patterns_sorted_all, Patterns_to_Index_all = pattern_analysis(patterns_all, graph_to_txhash_all, all_txs)
wt_summary_all.to_csv('../results/patterns/patterns_summary_all.csv', index = 0)






