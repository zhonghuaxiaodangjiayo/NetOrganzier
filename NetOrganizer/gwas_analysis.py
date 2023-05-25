from scipy.stats import binom_test
import pickle
from math import log10

import statistics
import openpyxl
from statsmodels.stats import multitest
import seaborn as sns
import os

import importdatas as datas
from sklearn.metrics.pairwise import cosine_similarity
from multiprocessing import Process, Queue, Manager
import random
import pickle
import heapq
from collections import OrderedDict
import numpy as np
import pandas as pd

from sklearn.metrics import precision_recall_curve

from sklearn import preprocessing

import matplotlib
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import gaussian_kde
import scipy.stats
from scipy.stats import levene

from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multitest import fdrcorrection

import csv
import json
from matplotlib_venn import venn2


def calculate(organSet, N, path):
    grn = datas.import_database_grnhg()
    geneORGANizer = datas.importgeneItem()
    organ_back = dict()
    # for organ in geneORGANizer['bodyparts']:
    #     organ_back[organ] = [len(geneORGANizer['bodyparts'][organ]), len(geneORGANizer['genes'])]
    '''更改:考虑在整个RE空间中organs的注释情况'''
    with open('/share/home/liangzhongming/NetOrganizer/RE-ORGANizer-master/generateREGOA/gwas_result/organ_RE_fmax-full10.txt', 'r') as f:
        for line in f:
            items = line.strip().split('\t')
            organ = items[0]
            cnt = 0
            for r in items[1:]:
                cnt += 1
            organ_back[organ] = [cnt, len(grn['res'])]

    organSetp = dict()
    for organ in organSet.keys():
        # print(f"organ : {organ}")
        # print(f"N : {N}")
        # print(f"{organSet[organ] / N} ;{organ_back[organ][0] / organ_back[organ][1]} \n")


        # 需要比对的是organ在背景RE中的注释比例，而不是organ在geneORGANizer中的注释比例
        if organSet[organ] / N >= organ_back[organ][0] / organ_back[organ][1]:
            '''
                binom_test(x, n, p)
                x:表示样本中符合条件
                n:样本总个体数:由GWAS性状的SNP转换为的RE数量
                p:表示在总体中符合条件的比例

                organSet[organ]:当前organ被SNP代表的RE注释的次数
                N:当前organ在整个RE空间上的注释比例
            '''
            organSetp[organ] = binom_test(organSet[organ], N, organ_back[organ][0] / organ_back[organ][1])

    organs = list(organSetp.keys())
    organs.sort(key=lambda x: organSetp[x])

    i = 0
    organSetq = dict()
    for organ in organs:
        i += 1
        p = organSetp[organ]
        organSetq[organ] = p * len(organ) / i

    organs.sort(key=lambda x: organSetq[x])
    
    
    g = open(path, 'w')
    key_name = path[96:]
    print(f"write:{key_name}")
    for organ in organs:
        if organSetp[organ] > 0.05:
            break
        # 依次写入：organ   在特定性状中注释次数     N   背景网络organ[0]   背景网络organ[1]
        g.write(organ)
        g.write('\t' + str(organSet[organ]) + '\t' + str(N) + '\t' + str(organ_back[organ][0] / organ_back[organ][1]))
        g.write('\t' + str(organSetp[organ]) + '\t')
        if organSetp[organ] != 0:
            g.write(str(-log10(organSetp[organ])))
        g.write('\t' + str(organSetq[organ]) + '\t')
        if organSetq[organ] != 0:
            g.write(str(-log10(organSetq[organ])))
        g.write('\n')
    g.close()
    return 0


def main():
    # g = open(dataset + 'MF' + 're-term-fmaxrawinter.pkl', 'rb+')
    # resmf = pickle.load(g)
    # g = open(dataset + 'MF' + '_backpro_fmaxrawinter.pkl', 'rb+')
    # go_backmf = pickle.load(g)

    # g = open(dataset + 'BP' + 're-term-fmaxrawinter.pkl', 'rb+')
    # resbp = pickle.load(g)
    # g = open(dataset + 'BP' + '_backpro_fmaxrawinter.pkl', 'rb+')
    # go_backbp = pickle.load(g)

    # g = open(dataset + 'CC' + 're-term-fmaxrawinter.pkl', 'rb+')
    # rescc = pickle.load(g)
    # g = open(dataset + 'CC' + '_backpro_fmaxrawinter.pkl', 'rb+')
    # go_backcc = pickle.load(g)

    # g = open(dataset + 'GRN.pkl', 'rb+')
    # grn = pickle.load(g)

    grn = datas.import_database_grnhg()
    res = dict()
    for re in grn['res']:
    # split line into chromosome number, start and end position
      chr, start, end = re.strip().split("_")

      # if the chromosome is not in the dictionary yet, initialize an empty dictionary
      if chr not in res: 
         res[chr] = {}

      # store start and end in the dictionary for the current re
      re = re.strip()
      res[chr][re] = {}
      res[chr][re]["start"] = int(start)
      res[chr][re]["end"] = int(end)
    
    # RE注释organs的字典:原始、-0.01、-0.015、-0.02、-0.03
    re_organ = dict()
    with open('/share/home/liangzhongming/NetOrganizer/RE-ORGANizer-master/generateREGOA/gwas_result/organ_RE_fmax-full10.txt', 'r') as f:
        for line in f:
            items = line.strip().split('\t')
            organ = items[0]
            for r in items[1:]:
                r_items = r.split('_')
                chr = r_items[0]
                
                if chr not in re_organ:
                    re_organ[chr] = {}
                if r not in re_organ[chr]:
                    re_organ[chr][r] = {'organs': []}
                
                re_organ[chr][r]['organs'].append(organ)


    # organSet = dict()
    inpath = '/share/home/liangzhongming/NetOrganizer/RE-ORGANizer-master/generateREGOA/GWAS_analysis/GWAS206/'
    outpath = '/share/home/liangzhongming/NetOrganizer/RE-ORGANizer-master/generateREGOA/GWAS_analysis/GWAS206_organ_full10/'
    # 获取文件夹中所有文件路径
    file_paths = [os.path.join(inpath, f) for f in os.listdir(inpath)]
    # 按照文件名进行排序
    file_paths_sorted = sorted(file_paths, key=lambda x: os.path.basename(x))
    M = len(file_paths_sorted)
    cnt_file = 0
    for filename in file_paths_sorted:
        organSet = dict()
        geneset = dict()
        key_name = filename[96:-4]
        cnt_file += 1
        print(f"handle file:{key_name}")
        print(f"{cnt_file} / {M}")
    # filename = "/share/home/liangzhongming/NetOrganizer/RE-ORGANizer-master/generateREGOA/GWAS_analysis/GWAS206/46_irnt.txt"
        N = 0
        cnt_continue = 0
        with open(filename, 'r') as f:
            for line in f:
                line = line.split()
                chr = line[0]
                pos = int(line[1])
                dismin = 99999999
                if not chr in res:
                    cnt_continue += 1
                    continue
                

                for re in res[chr].keys():
                    if res[chr][re]['end'] < pos:
                        dis = pos - res[chr][re]['end']
                    elif res[chr][re]['start'] > pos:
                        dis = res[chr][re]['start'] - pos
                    elif pos <= res[chr][re]['end'] and pos >= res[chr][re]['start']:
                        dismin = 0
                        cloest_re = re
                        break
                    if dis < dismin:
                        dismin = dis
                        cloest_re = re
                distances.append(dismin)
                if dismin < 1000:
                    N += 1 # 代表由SPN转换为的RE的数量:每个SNP选取最佳RE
                    if cloest_re in re_organ[chr]:
                        for organ in re_organ[chr][cloest_re]['organs']:
                            if not organ in organSet:
                                organSet[organ] = 0
                            organSet[organ] += 1
                    
                    for gene in grn['res'][cloest_re]['reg']:
                        if not gene in geneset:
                            geneset[gene] = 0
                        geneset[gene] += 1
        
        print(f"cnt_continue:{cnt_continue}")
        writePath = outpath + key_name
        calculate(organSet, N, writePath + '_result.txt')
        # calculate(gosetbp, go_backbp, N, resultpath + '_BP.txt')
        # calculate(gosetcc, go_backcc, N, resultpath + '_CC.txt')

        genes = list(geneset.keys())
        genes.sort(key=lambda x: geneset[x], reverse=True)
        with open(outpath + key_name + '_genes.txt', 'w') as f:
            for gene in genes:
                f.write(gene + '\t' + str(geneset[gene]) + '\n')

        organs = list(organSet.keys())
        organs.sort(key=lambda x: organSet[x], reverse=True)
        with open(outpath + key_name + '_organs.txt', 'w') as f:
            for organ in organs:
                f.write(organ + '\t' + str(organSet[organ]) + '\n')

        
    return 0


# def importgo(filename='./go.obo'):
#     return go


def save_distances():
    # with open('/share/home/liangzhongming/NetOrganizer/RE-ORGanizer-master/generateREGOA/gwas_result/distances.json', 'w') as f:
    #     json.dump(distances, f)

    with open('/share/home/liangzhongming/NetOrganizer/RE-ORGANizer-master/generateREGOA/gwas_result/distances.json') as f:
        distances = json.load(f)

    # 将距离值按所属区间计数
    interval = 1000
    counts, bins = np.histogram(distances, bins=range(0, 11000, interval))

    # 计算每个区间内数据数量占总数量的比例
    total_count = len(distances)
    proportions = [(count / total_count) for count in counts]

    # 绘制直方图
    fig, ax = plt.subplots()
    labels = [str(1000*i) for i in range(11)]
    positions = range(len(proportions))
    ax.bar(positions, proportions, width=1, align='edge')
    plt.xticks(positions, labels)
    ax.set_xlabel('min_distance to nearest RE')
    ax.set_ylabel('Peaks num')
    ax.set_title('Distribution of GWAS SNPs distance to nearest RE')
    plt.savefig('/share/home/liangzhongming/NetOrganizer/RE-ORGANizer-master/generateREGOA/gwas_result/peaks206.pdf')


def save_sim2graph():
    # 读取CSV文件
    df = pd.read_csv('/share/home/liangzhongming/NetOrganizer/RE-ORGANizer-master/generateREGOA/gwas_result/LDSC/simiLDSC206.csv', index_col=0)

    # 删除对角线元素
    for i in range(len(df)):
        df.iloc[i, i] = 0

    # 展平数据
    data = df.values.flatten()

    # 绘制直方图
    fig, ax = plt.subplots()
    ax.hist(data, bins=50, log=True)
    ax.set_xlabel('GWAS traits pair similarity JS')
    ax.set_ylabel('GWAS Traits pair num')
    ax.set_title('Distribution of GWAS Traits Pairs Similarity based on NetOrganizer')
    plt.savefig('/share/home/liangzhongming/NetOrganizer/RE-ORGANizer-master/generateREGOA/gwas_result/LDSC/GWAS_dis.pdf')


def find_pair():
    # 读取CSV文件
    with open('/share/home/liangzhongming/NetOrganizer/RE-ORGANizer-master/generateREGOA/gwas_result/base_full5/simi206full.csv', mode='r') as file:
        reader = csv.reader(file)
        rows = [row for row in reader]

    # 获取行列信息和相似度列表
    organs = rows[0][1:]  # 第一行除去第一个元素为列名，即器官名
    sims = [[float(val) for val in row[1:]] for row in rows[1:]]  # 从第二行开始为相似度

    # 找出相似度大于等于0.4的关系对
    pairs = []
    for i in range(len(organs)):
        for j in range(len(organs)):
            if i == j:
                continue  # 不考虑自身相似度
            if sims[i][j] >= 0.4:
                pair = (organs[i], organs[j], sims[i][j])
                if pair not in pairs and (pair[1], pair[0], pair[2]) not in pairs:
                    pairs.append(pair)

    # 将结果保存到本地txt文件
    with open('/share/home/liangzhongming/NetOrganizer/RE-ORGANizer-master/generateREGOA/gwas_result/base_full5/pair.txt', mode='w') as file:
        for pair in pairs:
            file.write(pair[0] + ' ' + pair[1] + ' ' + str(pair[2]) + '\n')


def find_com_diff():
    # 读取第一个txt文件
    with open('/share/home/liangzhongming/NetOrganizer/RE-ORGANizer-master/generateREGOA/gwas_result/base_full5/pair.txt', mode='r') as file:
        reader = csv.reader(file, delimiter=' ')
        data1 = set()
        for row in reader:
            if row and len(row) == 3:  # 确保格式正确
                pair = tuple(sorted([row[0], row[1]]))  # traitA traitB和traitB traitA视作一样
                data1.add((pair, float(row[2])))

    # 读取第二个txt文件
    with open('/share/home/liangzhongming/NetOrganizer/RE-ORGANizer-master/generateREGOA/gwas_result/LDSC/pair.txt', mode='r') as file:
        reader = csv.reader(file, delimiter=' ')
        data2 = set()
        for row in reader:
            if row and len(row) == 3:  # 确保格式正确
                pair = tuple(sorted([row[0], row[1]]))  # traitA traitB和traitB traitA视作一样
                data2.add((pair, float(row[2])))

    # 比对两个文件的数据，找出共同的和差异的trait pair
    shared_pairs = set()
    diff_pairs_1 = set()
    diff_pairs_2 = set()
    for pair, sim in data1:
        if pair in data2:
            shared_pairs.add(pair)
        else:
            diff_pairs_1.add(pair)
    for pair, sim in data2:
        if pair not in data1:
            diff_pairs_2.add(pair)

    # 将共同和差异的trait pair写入本地文件
    with open('/share/home/liangzhongming/NetOrganizer/RE-ORGANizer-master/generateREGOA/gwas_result/base_full5/shared_pairs.txt', mode='w') as file:
        for pair in shared_pairs:
            file.write(pair[0] + ' ' + pair[1] + ' ' + str(1.0) + '\n')
    with open('/share/home/liangzhongming/NetOrganizer/RE-ORGANizer-master/generateREGOA/gwas_result/base_full5/diff_pairs_our.txt', mode='w') as file:
        for pair in diff_pairs_1:
            file.write(pair[0] + ' ' + pair[1] + ' ' + str(1.0) + '\n')
    with open('/share/home/liangzhongming/NetOrganizer/RE-ORGANizer-master/generateREGOA/gwas_result/base_full5/diff_pairs_LDSC.txt', mode='w') as file:
        for pair in diff_pairs_2:
            file.write(pair[0] + ' ' + pair[1] + ' ' + str(1.0) + '\n')

    # 绘制文氏图
    venn2([data1, data2], set_colors=('#FF8080', '#80B2FF'), set_labels=['File1', 'File2'])
    plt.savefig('/share/home/liangzhongming/NetOrganizer/RE-ORGANizer-master/generateREGOA/gwas_result/base_full5/venn_diagram.png')
    # plt.show()

if __name__ == '__main__':
    # global go
    # go = importgo()
    # path = './inputfiledict/'
    # datapath = './datamgi/'
    # resultpath = './outputfiledict/'
    # count = 0
    # file = 'exampleinput.bed'

    # with open('./configures.txt','r') as f:
    #     sets = list()
    #     for line in f:
    #         sets.append(line.split()[1])
    #     path = sets[0]
    #     datapath = sets[1]
    #     resultpath = sets[2]
    #     file = sets[3]


    # global distances
    # distances = list()

    # main()
    save_distances()

    # save_sim2graph()
    # find_pair()
    # find_com_diff()

