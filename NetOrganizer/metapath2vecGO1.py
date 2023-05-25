
import gensim
import importdatas as datas
from random import choice
from gensim.models import Word2Vec, KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_recall_curve
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot,savefig
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
import seaborn as sns
import scipy.cluster.hierarchy as sch
import palettable
from random import shuffle
from multiprocessing import Process, Queue, Manager
import random
import pickle
import joblib
import calthres
import os

Function = 'CC'
path = './datamgi/'
notename = '-noDis'

# 导入gene_ORGANizer
# global gG
# global gR
# global gS
# g_O = datas.importgene_organs('./New_gene_organs_2.txt')
# with open(path+'diffpaths/6metapath_add_OGOG/g_O.txt', 'w') as f:
#     f.write(str(g_O))

# gG = datas.importgene_organs('./gene_germlayers.txt')
# gR = datas.importgene_organs('./gene_regions.txt')
# gS = datas.importgene_organs('./gene_systems.txt')
# print('import gene_organs(g_O) done')

# geneORGANizer = datas.importgeneORGANizer()
# with open(path + './geneORGANizer_result/gene_ORGANizer.txt', 'w') as f:
#     f.write(str(geneORGANizer))
# print('import geneORGANizer done')

geneORGANizer = datas.importgeneItem()
print(len(geneORGANizer['bodyparts']))
print(len(geneORGANizer['genes']))
print('import geneORGANizer done')


def preprocess():
    # 导入基因调控网络grn
    global grn
    # 人
    if path[6] == 'h':
        grn = datas.importgrnhg()    #    return {'genes':{gene:re,gene:re...},'res':{re:{'reg':genes,'bind':tfs}},'tfs':{tf:re,tf:re}}
    # 鼠鼠
    else:
        grn = datas.importgrnmgi()  # return {'genes':{gene:re,gene:re...},'res':{re:{'reg':genes,'bind':tfs}},'tfs':{tf:re,tf:re}}
    print('import grn done')

    # # 导入go数据库网络
    # global go
    # go = datas.importgo(path+'go.obo')      #    {go_id:{'is_a':terms,'children':terms,}}
    #
    # go = {x: go[x] for x in go if go[x]['namespace'] == Function}
    # print('import go done')

    # 注释操作：2022/04/26 in 15:05 因为test中无法访问到g_O
    # # 导入gene ORGANizer
    # global g_O
    # # global gG
    # # global gR
    # # global gS
    # g_O = datas.importgene_organs('./New_gene_organs_2.txt')
    # # gG = datas.importgene_organs('./gene_germlayers.txt')
    # # gR = datas.importgene_organs('./gene_regions.txt')
    # # gS = datas.importgene_organs('./gene_systems.txt')
    # print('import gene_organs(g_O) done')

    # # 生成goa和goaraw网络
    # global goa
    # global goaraw
    # goaraw = datas.importgoa(go, path+'train.txt')    #    return {'genes':{gene:terms,...},'terms':{term:genes}:}
    # print('import goa done')
    # goa=datas.goatpr(goaraw, go)
    # print('tpr goa done')

    # 导入蛋白互作网络PPI
    global ppi
    ppi = datas.importppi(path+'ppifull.txt')
    print('import ppi done')

    print('genes num in geneORGANizer:', len(set(geneORGANizer['genes'])), 'bodyparts num in geneORGANizer:', len(geneORGANizer['bodyparts']))
    print('genes num in grn:', len(set(grn['genes'])))
    print('genes num in ppi:', len(set(ppi.keys())))
    # print('terms num in go:', len(go))


    # global gene_n_goa
    # gene_n_goa = grn['genes'].keys()-goa['genes'].keys()  # grn中存在且goa中不存在的gene
    # print('gene in grn not in goa:', len(gene_n_goa))

    genetf = set(grn['genes'].keys()) | set(grn['tfs'].keys())

    # gene_n_grn = goa['genes'].keys()-genetf  # goa中存在且grn中不存在的gene
    # print('gene in goa not in grn:', len(gene_n_grn))

    print('intersection genes of grn_geneORGANizer_ppi num:', len(set(grn['genes'].keys()) & set(geneORGANizer['genes'].keys()) & set(ppi.keys())))
    print('res num:', len(grn['res']))
    print('tfs num:', len(grn['tfs']))
    # print(gO['genes'].keys())
    # print(set(grn['genes'].keys()) & set(gO['genes'].keys()) & set(ppi.keys()))


def pickppi(ppisp):
    x = random.randint(0, sum(list(ppisp.values()))-1)
    count = 0
    for item in ppisp.keys():
        count += ppisp[item]
        if count > x:
            return item
    return item

# 从gene开始的游走
def genwalk(walks, genesets, walknum, walklen, geneORGANizer, logo):
    # 原始函数：def genwalk(walks, genesets, walknum, walklen, goa, go, ppi, logo):
    # 因为有些metapath无法迁移，用不到go、goa、ppi
    # g_O_genes = set(g_O['genes'].keys())
    g_O_organs = set(geneORGANizer['bodyparts'].keys())
    M = len(genesets)
    count = 0
    for gene in genesets:  # 对于每个基因，走了20条拥有50个结点的GBGB...
        for i in range(walknum*10):  # 每个gene节点走20条序列  # 50 ————> 50 *10
            # walk = []                                  # GTTTTTTT...(无法迁移)
            # #  将当前gene纳入walk
            # walk.append(gene)
            # # 将当前gene在goa网络中对应的term纳入walk
            # term = choice(list(goa['genes'][gene]))
            # walk.append(term)
            # # 将go网络中，与该term有is_a、children关系的term纳入walk
            # for j in range(walklen):
            #     term = choice(list(go[term]['is_a'] | go[term]['children']))
            #
            #     walk.append(term)
            # walks.append(walk)

            walk = []                                #GTGTGTGTGT...——> GOGOGOGO...
            walk.append(gene)
            bodypart = choice(list(geneORGANizer['genes'][gene]))  # 随机选取一个和当前基因有关的组织
            walk.append(bodypart)
            # 同上一条元路径，先纳入gene和其对应term，
            # 对于当前term，将goa网络中，与其对应的gene纳入walk
            # 反复操作
            for j in range(walklen):  # 每条序列走50个节点
                gene = choice(list(geneORGANizer['bodyparts'][bodypart]))
                walk.append(gene)
                bodypart = choice(list(geneORGANizer['genes'][gene]))
                walk.append(bodypart)
            walks.append(walk)



            # walk = []                                 # GTTGGTTGGTTG...(无法迁移)
            # # 纳入当前gene
            # walk.append(gene)
            # # 选择goa网络中当前基因对应的term，同时该term还需满足存在于GOAterms中
            # term = choice(list(goa['genes'][gene] & GOAterms))
            # walk.append(term)
            # # 选择与当前term存在is_a或者children关系的term，且该term还需满足存在于GOAterms中
            # tterm = choice(list((go[term]['is_a'] | go[term]['children']) & GOAterms))
            #
            # # 若该tterm在goa中不存在，且在GOAgenes中存在，且在ppi网络中存在，则纳入walk，并且开启下一次循环
            # if not goa['terms'][tterm] & GOAgenes & ppi.keys():
            #     walk.append(tterm)
            #     walks.append(walk)
            #     continue
            # # 否则该tterm纳入walk后，继续本轮循环
            # term = tterm
            # walk.append(term)
            #
            # # 选择goa网络中当前term(为之前的tterm)对应的gene，且满足存在于GOAgenes中，且满足存在于ppi网络中
            # tgene = choice(list(goa['terms'][term] & GOAgenes & ppi.keys()))
            # # 若PPI中该tgene的key不存在，且其在GOAgenes中存在，则纳入walk作为结尾，纳入walks，结束本轮循环
            # if not ppi[tgene].keys() & GOAgenes:
            #     walk.append(tgene)
            #     walks.append(walk)
            #     continue
            # # 否则纳入walk后继续循环
            # gene = tgene
            # walk.append(gene)
            #
            # for j in range(walklen // 2):  # //为取整除，保留整数部分
            #     gene = pickppi({x:ppi[gene][x] for x in ppi[gene].keys() if x in GOAgenes})
            #     walk.append(gene)
            #     term = choice(list(goa['genes'][gene] & GOAterms))
            #     walk.append(term)
            #
            #     tterm = choice(list((go[term]['is_a'] | go[term]['children']) & GOAterms))
            #     if not goa['terms'][tterm] & GOAgenes & ppi.keys():
            #         walk.append(tterm)
            #         break
            #     term = tterm
            #     walk.append(term)
            #
            #     tgene = choice(list(goa['terms'][term] & GOAgenes & ppi.keys()))
            #     if not ppi[tgene].keys() & GOAgenes:
            #         walk.append(tgene)
            #         break
            #     gene = tgene
            #     walk.append(gene)
            #
            #     #print(len(walk), walk)
            # walks.append(walk)

        count += 1
        if count % 10 == 0:
            print(logo, count, '/gene', M)

    # # 2022/04/28 in 1:24 返厂改进：没有定义一条从组织开始跑的metapath，且组织之间没有边，导致某些组织没有出现在游走序列中，得不到低维向量表示
    # count = 0
    # M = len(g_O['organs'].keys())
    # for organ in g_O_organs:
    #     walk = []  # GTGTGTGTGT...——> GOGOGOGO...——> OGOGOGOG 很关键
    #     walk.append(organ)
    #     gene = choice(list(g_O['organs'][organ]))  # 随机选取一个和当前组织有关的基因
    #     walk.append(gene)
    #     # 同上一条元路径，先纳入gene和其对应term，
    #     # 对于当前term，将goa网络中，与其对应的gene纳入walk
    #     # 反复操作
    #     for j in range(walklen):
    #         organ = choice(list(g_O['genes'][gene]))
    #         walk.append(organ)
    #         gene = choice(list(g_O['organs'][organ]))
    #         walk.append(gene)
    #     walks.append(walk)
    #
    #     count += 1
    #     if count % 10 == 0:
    #         print(logo, count, '/organ', M)


# 这是怎么游走的？ walknum * 5和walklen // 4目的
def grnwalk(walks, genesets, walknum, walklen, grn, logo):
    M = len(genesets)  # GRTRG
    count = 0
    for gene in genesets:
        for i in range(walknum):  # 每个gene拥有20*5条 G-RE-TF-RE-TF...的序列  这个20*5是调参出来的吗？ # 100 *2
            walk = []
            walk.append(gene)
            # 将在grn中与gene有关的re、tf加入，再将与tf有关的re加入，再将与该re有关的tf加入    #gene-re-tf-re-tf
            for j in range(walklen // 4):  # 每个序列格式G-RE-TF-RE-TF-RE-TF...
                re = choice(list(grn['genes'][gene]))
                walk.append(re)
                tf = choice(list(grn['res'][re]['binded']))
                walk.append(tf)
                re = choice(list(grn['tfs'][tf]))
                walk.append(re)
                tf = choice(list(grn['res'][re]['reg']))   # 师姐写错了 该tf就是Gene
                walk.append(tf)
            walks.append(walk)
        count += 1
        if count % 10 == 0:
            print(logo, 'No.8', count, 'grn', M)

def mp2vwalk(walknum,walklen):
    global walks
    walks = []
    # gOgenes = set(g_O['genes'].keys())
    # g_O_organs = set(g_O['organs'].keys())

    # gO_organs = set(geneORGANizer['organs'].keys())
    # gO_regions = set(geneORGANizer['regions'].keys())
    # gO_systems = set(geneORGANizer['systems'].keys())
    # gO_germlayers = set(geneORGANizer['germlayers'].keys())

    gO_genes = set(geneORGANizer['genes'].keys())
    gO_bodyparts = set(geneORGANizer['bodyparts'].keys())

    '''
    L=len(grn['res'].keys())          #add re into goa   RE-Term
    count=0
    for re in grn['res'].keys():
        count+=1
        if count%10000==0:
            print(count,'/resgoa',L)
        for gene in (grn['res'][re]['reg']) & GOAgenes:
            for term in goaraw['genes'][gene]:
                walks.append([re,term])

    walks = walks*10
    print(len(walks))
    '''
    # 新增metapath  re-bodypart 2022/05/17
    # without re-bodypart看效果 2022/05/19
    L = len(grn['res'].keys())
    count = 0
    for re in grn['res'].keys():
        count += 1
        if count % 1000 == 0:
            print(count, '/res_geneORGANizer', L)
        for gene in (grn['res'][re]['reg']) & geneORGANizer['genes'].keys():
            for bodypart in geneORGANizer['genes'][gene]:
                walks.append([re, bodypart])
    walks = walks * 40



    # 这个re-goa.pkl是从哪来的？ ：之前运行好了存储起来的，直接取出就可以用，不用考虑格式
    # 在reannotate.py中出入.pkl文件
    # 在此处，将re-goa文件中的re-term写入walks是为什么？
    # 注释操作：2022/04/24 23:23

    # g = open(path + Function + 're-goa.pkl', 'rb+')
    # res = pickle.load(g)
    # for re in res.keys():
    #     for term in res[re]:
    #         walks.append([re, term])
    # walks = walks * 40     # 这是不是en调参？将re-term的贡献拉满
    # print(len(walks))


    tfs = grn['tfs'].keys() & grn['genes'].keys()
    L = len(tfs)
    count = 0
    for tf in tfs:
        count += 1
        print(count, '/ tfs', L)
        for i in range(walknum):
            walk = [tf]
            ttf = tf
            for j in range(walklen):      # TRTRTRT...(TF-RE-TF-RE...)
                tre = choice(list(grn['genes'][ttf]))
                walk.append(tre)
                if list(grn['res'][tre]['binded'] & tfs):
                    ttf = choice(list(grn['res'][tre]['binded'] & tfs))
                    walk.append(ttf)
                else:
                    break
            walks.append(walk)

    # 2022/04/28 in 1:24 返厂改进：没有定义一条从组织开始跑的metapath，且组织之间没有边，导致某些组织没有出现在游走序列中，得不到低维向量表示
    # 2022/05/12 in 10:35 更新至geneORGANizer版本
    # 2022/05/16 in 14:23 更新至geneItem版本
    # 2022/05/19 in 12:45 注释BG看效果
    count = 0
    M = len(geneORGANizer['bodyparts'].keys())
    for bodypart in gO_bodyparts:
        # with open(path + 'diffpaths/6metapath_add_OGOG/seq_start_with_organ.txt', 'a+') as f:
        #     # f.write('cur organ :', organ) 错误写法
        #     f.write('cur organ :' + str(organ) + '\t')
        walk = []  # GTGTGTGTGT...——> GBGBGB...——> BGBGBG... 很关键
        walk.append(bodypart)

        # with open(path + 'diffpaths/6metapath_add_OGOG/seq_start_with_organ.txt', 'a+') as f:
        #     f.write('cur organ can choice genes:' + str(list(g_O['organs'][organ])))

        gene = choice(list(geneORGANizer['bodyparts'][bodypart]))  # 随机选取一个和当前bodypart有关的基因
        # with open(path + 'diffpaths/6metapath_add_OGOG/seq_start_with_organ.txt', 'a+') as f:
        #     f.write('organ :' + str(organ) + 'choice gene:' + str(gene))

        walk.append(gene)
        # 同上一条元路径，先纳入gene和其对应term，
        # 对于当前term，将goa网络中，与其对应的gene纳入walk
        # 反复操作
        for j in range(walklen):                                        # 最终walk：B-G-B-G
            bodypart = choice(list(geneORGANizer['genes'][gene]))
            walk.append(bodypart)
            gene = choice(list(geneORGANizer['bodyparts'][bodypart]))
            walk.append(gene)
        walks.append(walk)

        count += 1
        if count % 10 == 0:
            print(count, '/bodypart', M)

    L = len(grn['res'].keys())
    count = 0
    for re in grn['res'].keys():        #RGGR RGGR RGGR...iii
        count += 1
        if count % 100 == 0:
            print(count, '/res', L)
        walk = [re]
        tre = re
        for j in range(walklen//4):  #  走满50个节点作为一条序列 该序列中元素都是RGGR RGGR
            if not list(grn['res'][tre]['reg'] & ppi.keys()):
                break
            tgene = choice(list(grn['res'][tre]['reg'] & ppi.keys()))
            walk.append(tgene)
            if not list(ppi[tgene].keys() & grn['genes'].keys()):
                break
            tgene = choice(list(ppi[tgene].keys() & grn['genes'].keys()))
            walk.append(tgene)
            tre = choice(list(grn['genes'][tgene]))
            walk.append(tre)
            tre = choice(list(grn['genes'][tgene]))
            walk.append(tre)
        walks.append(walk)
    print(len(walks))

    # 用多进程跑  #  2022/05/18 16:11 删除gene-bodypart...测试性能变化
    walksgene = Manager().list()
    N = 8
    geneset = []
    for i in range(N):
        geneset.append(set())
    for i in geneORGANizer['genes'].keys():
        geneset[random.randint(0, N - 1)].add(i)
    processes = []
    for i in range(N):
        processes.append(Process(target=genwalk, args=(walksgene, geneset[i], walknum, walklen, geneORGANizer, 'p'+str(i))))
    for i in range(N):
        processes[i].start()
    for i in range(N):
        processes[i].join()
    walks = walks+list(walksgene)
    print(len(walks))


    M = len(ppi.keys())              #GGGGGGG (PPI) viii
    count = 0
    for gene in ppi.keys():
        count += 1
        for i in range(walknum):
            walk = []
            walk.append(gene)
            for j in range(walklen):
                gene = pickppi(ppi[gene])
                walk.append(gene)
            walks.append(walk)
        if count % 100 == 0:
            print(count, '/ppi', M)


    walksgrn = Manager().list()
    N = 8
    geneset = []
    for i in range(N):
        geneset.append(set())
    for i in grn['genes'].keys():
        geneset[random.randint(0, N - 1)].add(i)
    processes = []
    for i in range(N):         # GRTRG iv
        processes.append(
            Process(target=grnwalk, args=(walksgrn, geneset[i], walknum, walklen, grn, 'p' + str(i))))
    for i in range(N):
        processes[i].start()
    for i in range(N):
        processes[i].join()
    walks = walks + list(walksgrn)
    print(len(walks))
    


    # N = len(grn['genes'].keys())                    #GRTRT...RT  是GRTRG 已经放入多进程跑
    # count = 0
    # for gene in grn['genes'].keys():
    #     for i in range(walknum*5):
    #         walk = []
    #         walk.append(gene)
    #         for j in range(walklen//4):
    #             re = choice(list(grn['genes'][gene]))
    #             walk.append(re)
    #             tf = choice(list(grn['res'][re]['binded']))
    #             walk.append(tf)
    #             re = choice(list(grn['tfs'][tf]))
    #             walk.append(re)
    #             gene = choice(list(grn['res'][re]['reg']))
    #             walk.append(gene)
    #         walks.append(walk)
    #     count += 1
    #     if count % 100 == 0:
    #         print(count, 'grn', N)

    return walks

def lzm_mp2walk(walknum, walklen):
    global walks
    walks = []

    # # 6. TF-RE-G-B-G-RE-TF   2022/05/25 15:10更改：因为RE-G-B条件较苛刻，需要PPI进行过渡，更改为RE-G-G-B
    # # 6. TF-RE-G-G-B-G-G-RE-TF
    # L = len(grn['tfs'].keys())
    # count = 0
    # real_nums_of_TFpath = 0
    # count_pass_gene = 0
    # # 每个tf采样一次，如果该路径走不通，break换下一个tf走，所以最终该元路径会比较少
    # # 针对措施：将最终采样到的该路径进行简单数乘
    # for tf in grn['tfs'].keys():  # 每个tf采样一次，如果该路径走不通，break换下一个tf走，所以最终该元路径会比较少
    #     count += 1
    #     if count % 10 == 0:
    #         print(count, '/No.6_TFs', L)
    #
    #     walk = []
    #     walk.append(tf)
    #
    #     for j in range(walklen//8):
    #         re = choice(list(grn['tfs'][tf]))
    #         walk.append(re)
    #         '''
    #         if not list(grn['res'][re]['reg'] & geneORGANizer['genes'].keys()):
    #             count_pass_gene += 1
    #             print(count_pass_gene)
    #             break
    #         print(list(grn['res'][re]['reg']))  # out:多只有一个基因，因为re调控的基因数量不会太多
    #         print(list(grn['res'][re]))  # out:['reg', 'binded']
    #         # 取交集之后很可能变为空集
    #         gene = choice(list(grn['res'][re]['reg'] & geneORGANizer['genes'].keys()))
    #         walk.append(gene)
    #         '''
    #         if not list(grn['res'][re]['reg'] & ppi.keys()):
    #             break
    #         gene = choice(list(grn['res'][re]['reg'] & ppi.keys()))
    #         walk.append(gene)
    #
    #         # print(ppi[gene])
    #         # print(ppi.keys())
    #         if not list(geneORGANizer['genes'].keys() & ppi[gene].keys()):
    #             break
    #         tgene = choice(list(ppi[gene].keys() & geneORGANizer['genes'].keys()))
    #         walk.append(tgene)
    #
    #         bodypart = choice(list(geneORGANizer['genes'][tgene]))
    #         walk.append(bodypart)
    #
    #         # if not list(geneORGANizer['bodyparts'][bodypart] & grn['genes'].keys()):
    #         #     count_pass_gene += 1
    #         #     print(count_pass_gene)
    #         #     break
    #         gene = choice(list(geneORGANizer['bodyparts'][bodypart] & ppi.keys()))
    #         walk.append(gene)
    #
    #         tgene = choice(list(grn['genes'].keys() & ppi[gene].keys()))
    #         walk.append(tgene)
    #
    #         # print(grn['genes'][gene])  # out:re
    #         re = choice(list(grn['genes'][tgene]))
    #         walk.append(re)
    #
    #         tf = choice(list(grn['res'][re]['binded']))
    #         walk.append(tf)
    #     # print('curren tf walk : ', walk)
    #     walks.append(walk)
    #
    # # print('real_nums_of_TFpath : ', real_nums_of_TFpath)
    # print(len(walks))
    # walks = walks * 500  # 50*10 = 500
    # print(len(walks))
    # #
    # # 5. TF-RE-G-RE-TF
    # L = len(grn['tfs'].keys())
    # count = 0
    # subwalk = []
    # for tf in grn['tfs'].keys():  # 每个tf采样一次，如果该路径走不通，break换下一个tf走，所以最终该元路径会比较少
    #     count += 1
    #     if count % 10 == 0:
    #         print(count, '/No.5_TFs', L)
    #
    #     walk = []
    #     walk.append(tf)
    #
    #     for j in range(walklen // 4):
    #         re = choice(list(grn['tfs'][tf]))
    #         walk.append(re)
    #         '''
    #         if not list(grn['res'][re]['reg'] & geneORGANizer['genes'].keys()):
    #             count_pass_gene += 1
    #             print(count_pass_gene)
    #             break
    #         print(list(grn['res'][re]['reg']))  # out:多只有一个基因，因为re调控的基因数量不会太多
    #         print(list(grn['res'][re]))  # out:['reg', 'binded']
    #         # 取交集之后很可能变为空集
    #         gene = choice(list(grn['res'][re]['reg'] & geneORGANizer['genes'].keys()))
    #         walk.append(gene)
    #         '''
    #         if not list(grn['res'][re]['reg'] & ppi.keys()):
    #             break
    #         gene = choice(list(grn['res'][re]['reg'] & ppi.keys()))
    #         walk.append(gene)
    #
    #         re = choice(list(grn['genes'][gene]))
    #         walk.append(re)
    #
    #         tf = choice((list(grn['res'][re]['binded'])))
    #         walk.append(tf)
    #     subwalk += walk
    #
    #
    # print(len(subwalk))
    # subwalk = subwalk * 500  # 500 ————> 1000
    # print(len(subwalk))
    # walks.append(subwalk)
    # #
    # 3. G-B-G...
    walksgene = Manager().list()
    N = 8
    geneset = []
    for i in range(N):
        geneset.append(set())
    for i in geneORGANizer['genes'].keys():
        geneset[random.randint(0, N - 1)].add(i)
    processes = []
    for i in range(N):
        processes.append(Process(target=genwalk, args=(walksgene, geneset[i], walknum, walklen, geneORGANizer, 'p'+str(i))))
    for i in range(N):
        processes[i].start()
    for i in range(N):
        processes[i].join()
    walks = walks+list(walksgene)
    print(len(walks))
    # #
    # #
    # #
    # # 1. TF-RE-TF-RE...
    # tfs = grn['tfs'].keys() & grn['genes'].keys()
    # L = len(tfs)
    # count = 0
    # for tf in tfs:
    #     subwalk = []
    #     count += 1
    #     print(count, '/ No.1_tfs', L)
    #     for i in range(walknum):
    #         walk = [tf]
    #         ttf = tf
    #         for j in range(walklen):      # TRTRTRT...(TF-RE-TF-RE...)
    #             tre = choice(list(grn['genes'][ttf]))
    #             walk.append(tre)
    #             if list(grn['res'][tre]['binded'] & tfs):
    #                 ttf = choice(list(grn['res'][tre]['binded'] & tfs))
    #                 walk.append(ttf)
    #             else:
    #                 break
    #         subwalk += walk
    #     subwalk = subwalk*10  # 100
    #     walks.append(subwalk)
    # # with open('./process_result/TF_RE_TF.txt', 'w') as f:
    # #     f.write(str(walks))
    # #
    # #
    # # 2. RE-G-RE...
    # L = len(grn['res'].keys())
    # count = 0
    # subwalk = []
    # for re in grn['res'].keys():
    #     count += 1
    #     if count % 100 == 0:
    #         print(count, '/No.2_res', L)
    #     walk = [re]
    #     tre = re
    #     for j in range(walklen//4):  #  走满50个节点作为一条序列
    #         if not list(grn['res'][tre]['reg']):
    #             break
    #         tgene = choice(list(grn['res'][tre]['reg']))
    #         walk.append(tgene)
    #
    #         tre = choice(list(grn['genes'][tgene]))
    #         walk.append(tre)
    #     # walk = walk * 2  # 16w ————> 32w
    #     walks.append(walk)
    # print(len(walks))
    # #
    # #
    # # 4. RE-G-B-G-RE...
    # L = len(grn['res'].keys())
    # count = 0
    # for re in grn['res'].keys():
    #     count += 1
    #     if count % 100 == 0:
    #         print(count, '/No.4_res_REGBGRE', L)
    #     walk = [re]
    #     tre = re
    #     for j in range(walklen // 4):
    #         if not list(grn['res'][tre]['reg'] & geneORGANizer['genes'].keys()):
    #             break;
    #         tgene = choice(list(grn['res'][tre]['reg'] & geneORGANizer['genes'].keys()))
    #         walk.append(tgene)
    #
    #         bodypart = choice(list(geneORGANizer['genes'][tgene]))
    #         walk.append(bodypart)
    #
    #         tgene = choice(list(grn['genes'].keys() & geneORGANizer['bodyparts'][bodypart]))
    #         walk.append(tgene)
    #
    #         re = choice(list(grn['genes'][tgene]))
    #         walk.append(re)
    #     # walk = walk * 2
    #     walks.append(walk)
    # print(len(walks))
    # #
    # #
    # #
    # # 7. G-G-G...
    # M = len(ppi.keys())              #GGGGGGG (PPI) viii
    # count = 0
    # for gene in ppi.keys():
    #     count += 1
    #     for i in range((walknum//10)):  # //5 ————> //3————> //10
    #         walk = []
    #         walk.append(gene)
    #         for j in range(walklen):
    #             gene = pickppi(ppi[gene])
    #             walk.append(gene)
    #         walks.append(walk)
    #     if count % 100 == 0:
    #         print(count, '/No.7_ppi', M)
    # #
    # # 8. GFRFG
    # # walksgrn = Manager().list()
    # # N = 8
    # # geneset = []
    # # for i in range(N):
    # #     geneset.append(set())
    # # for i in grn['genes'].keys():
    # #     geneset[random.randint(0, N - 1)].add(i)
    # # processes = []
    # # for i in range(N):         # GRTRG iv
    # #     processes.append(
    # #         Process(target=grnwalk, args=(walksgrn, geneset[i], walknum, walklen, grn, 'p' + str(i))))
    # # for i in range(N):
    # #     processes[i].start()
    # # for i in range(N):
    # #     processes[i].join()
    # # walks = walks + list(walksgrn)
    # # print(len(walks))
    # #
    # # 9.RGGR...
    # L = len(grn['res'].keys())
    # count = 0
    # for re in grn['res'].keys():  # RGGR RGGR RGGR...iii
    #     count += 1
    #     if count % 100 == 0:
    #         print(count, '/res', L)
    #     walk = [re]
    #     tre = re
    #     for j in range(walklen // 4):  # 走满50个节点作为一条序列 该序列中元素都是RGGR RGGR
    #         if not list(grn['res'][tre]['reg'] & ppi.keys()):
    #             break
    #         tgene = choice(list(grn['res'][tre]['reg'] & ppi.keys()))
    #         walk.append(tgene)
    #         if not list(ppi[tgene].keys() & grn['genes'].keys()):
    #             break
    #         tgene = choice(list(ppi[tgene].keys() & grn['genes'].keys()))
    #         walk.append(tgene)
    #         tre = choice(list(grn['genes'][tgene]))
    #         walk.append(tre)
    #         tre = choice(list(grn['genes'][tgene]))
    #         walk.append(tre)
    #     walks.append(walk)
    # print(len(walks))
    #
    # # 10.RB...
    # # L = len(grn['res'].keys())
    # # count = 0
    # # for re in grn['res'].keys():
    # #     count += 1
    # #     if count % 1000 == 0:
    # #         print(count, '/NO.10 res_geneORGANizer', L)
    # #     for gene in (grn['res'][re]['reg']) & geneORGANizer['genes'].keys():
    # #         for bodypart in geneORGANizer['genes'][gene]:
    # #             for t in range(40):
    # #                 walks.append([re, bodypart])
    return walks


def main():


    # walknum=20  # 从每个节点开始游走出walknum条路径
    # walklen=50  # 每条路径50个节点
    # walks=mp2vwalk(walknum, walklen)

    # walknum=50  # 从每个节点开始游走出walknum条路径
    # walklen=100  # 每条路径walklen个节点
    # walks=lzm_mp2walk(walknum, walklen)
    #
    # global model
    #
    #
    # f = open('./lzm_mp1_result/' + 'joblib_GB_180w_geneORGANizer_metapath' + '_walks' + notename + '.pkl', 'wb')
    # joblib.dump(walks, f)
    # f.close()
    #
    # f = open('./lzm_mp1_result/' + 'pickle_GB_180w_geneORGANizer_metapath' + '_walks' + notename + '.pkl', 'wb')
    # pickle.dump(walks, f)
    # f.close()

    # f = open('./lzm_mp1_result/' + 'onlyGB170w_geneORGANizer_metapath_ab' + '_walks' + notename + '.pkl', 'ab')
    # pickle.dump(walks, f)
    # f.close()
    #
    # f = open('./lzm_mp1_result/' + 'GB_85w_geneORGANizer_metapath' + '_walks' + notename + '.txt', 'a')
    # f.write(str(walks))
    # f.close()


    g = open('./lzm_mp1_result/' + 'onlyGB170w_geneORGANizer_metapath' + '_walks' + notename + '.pkl', 'rb+')
    walks = pickle.load(g)

    print('walk done')
    # 完整metapath
    # with open('./process_result/' + '5_20_0026_full7path_geneORGANizer_metapath' + '_walks.txt', 'w') as f:
    #     f.write(str(walks))

    # model = Word2Vec(walks, vector_size=128, window=2, min_count=0, sg=1, workers=16, epochs=10)
    # model = Word2Vec(walks, vector_size=128, window=2, min_count=0, sg=1, workers=16, epochs=10)  # 0.78
    # model = Word2Vec(walks, vector_size=150, window=5, min_count=0, sg=1, workers=16, epochs=10)  # 0.69
    # model = Word2Vec(walks, vector_size=150, window=2, min_count=0, sg=1, workers=16, epochs=10)  # 0.79
    # model = Word2Vec(walks, vector_size=200, window=2, min_count=0, sg=1, workers=16, epochs=10)  # 0.7938
    # model = Word2Vec(walks, vector_size=200, window=3, min_count=0, sg=0, workers=16, epochs=10)  # 0.56
    # model = Word2Vec(walks, vector_size=200, window=3, min_count=0, sg=1, workers=16, epochs=10)  # 0.75

    # lzm_mp1
    # model = Word2Vec(walks, vector_size=200, window=8, min_count=0, sg=1, workers=16, epochs=10)  # 5_25_1500_full7path
    # model = Word2Vec(walks, vector_size=200, window=8, min_count=0, sg=1, workers=16, epochs=10)  # 0.7

    model = Word2Vec(walks, vector_size=200, window=2, min_count=0, sg=1, workers=16, epochs=10)
    print('train done')

    model.save('./lzm_mp1_result/' + 'GB170w_sz2_vec')
    model.wv.save_word2vec_format('./lzm_mp1_result/' + 'GB170w_sz2_vec.txt')

    # 再训练
    # model = gensim.models.Word2Vec.load('./lzm_mp1_result/' + 'RB+GB+GRTRG_vec')
    # # model.build_vocab(walks)
    # model.build_vocab(walks, update=True)
    # model.train(walks, total_examples=model.corpus_count, epochs=10)
    # print('train done')
    #
    # # model.save('./lzm_mp1_result/' + 'RB+GB+GRTRG_vec')
    # # model.wv.save_word2vec_format('./lzm_mp1_result/' + 'RB+GB+GRTRG_vec.txt')
    #
    # model.save('./lzm_mp1_result/' + '5_30_1554_10path_mean16w_up3_sz8_vec')
    # model.wv.save_word2vec_format('./lzm_mp1_result/' + '5_30_1554_10path_mean16w_up3_sz8_vec.txt')


def importvec(filename):
    vecs = dict()
    with open(filename, 'r') as f:
        next(f)
        for line in f:
            line = line.split()
            obj = line[0]
            for index in range(1, len(line)):  # organs名字可能是'xxx xxx xxx'形式
                if line[index].isalpha():
                    obj += ' '
                    obj += line[index]
                else:
                    break
            vecs[obj] = [float(x) for x in line[index:]]
    #  测试是否正确拿到organ名称
    # with open('./process_result/lzm_test_vecs.txt', 'w') as f:
    #     f.write(str(vecs))
    print('import vecs done')
    return vecs

def getauroc(y, prob):
    fpr, tpr, threshold = roc_curve(y, prob)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值
    lw = 1
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC of metapath2vec in GO&GRN')
    plt.legend(loc="lower right")
    plt.savefig("./lzm_mp1_result/aucs/GO&GRN_RB170w_sz2" + ".jpg")
    return roc_auc

def test(vecs):
    '''
    standard = []
    score = []
    with open(path+'test.txt', 'r') as f:
        count = 0
        for line in f:
            line = line.split()
            if line[0] in vecs.keys() and line[1] in vecs.keys() & gO.keys(): # 如果gene存在vecs文件中，且term存在vecs文件和go文件中
                #ans=max({model.wv.similarity(line[0],x) for x in go[line[1]]['descent']&set(vecs.keys())}|{model.wv.similarity(line[0],line[1])})
                ans = max({cosine_similarity([vecs[line[0]], vecs[x]])[0][1] for x in go[line[1]]['descent'] & set(vecs.keys())} | {cosine_similarity([vecs[line[0]], vecs[line[1]]])[0][1]})
                # gene和term算余弦相似度 | gene和当前term的descent算余弦相似度  取其中的最大值
                score.append(ans)
                standard.append(1)
                count += 1
                if count % 100 == 0:
                    print(count, 'in 26824 pos')

    with open(path+'neg.txt', 'r') as f:
        count = 0
        for line in f:
            line = line.split()
            if line[0] in vecs.keys() and line[1] in vecs.keys()& go.keys():
                #ans=max({model.wv.similarity(line[0],x) for x in go[line[1]]['descent']&set(vecs.keys())}|{model.wv.similarity(line[0],line[1])})
                ans = max({cosine_similarity([vecs[line[0]], vecs[x]])[0][1] for x in go[line[1]]['descent']&set(vecs.keys())}|{cosine_similarity([vecs[line[0]],vecs[line[1]]])[0][1]})
                score.append(ans)
                standard.append(0)
                count += 1
                if count % 1000 == 0:
                    print(count, 'in 290000 neg')
    '''

    # 仿照以上，计算gene和organ余弦相似度
    # print('import g_O done(because gloable g_O is not work)')
    global geneORGANizer
    # print('g_O.keys() :', g_O.keys())
    standard = []  # 空字典是{}  空集合set()  空列表[]或list()
    score = []
    with open('./process_result/test5.txt', 'r') as f:
        print('open test.txt done')
        count = 0
        pass_count = 0
        line_count = 1
        for line in f:
            line = line.split()

            bodypart = line[1]
            for index in range(2, len(line)):
                if line[index].isalpha():
                    bodypart += ' '
                    bodypart += line[index]
                else:
                    break

            # bodypart = ' '.join(bodypart)
            # if bodypart in vecs.keys():
            #     print('bodypart in vecs.keys()')
            # else:
            #     print('not in : ')
            #     print(bodypart)

            if line[0] in vecs.keys() and bodypart in vecs.keys() & geneORGANizer['bodyparts'].keys():  # 如果gene存在vecs文件中，且bodypart存在vecs文件和geneORGANizer文件中
                ans = {cosine_similarity([vecs[line[0]], vecs[bodypart]])[0][1]}
                # print('test : ')
                # print('before : ', ans)
                # # ans = ans.toArray()[0]
                # # ans = ans.iterator().next()  # AttributeError: 'set' object has no attribute 'iterator'
                ans = list(ans)[0]
                # print('after : ', ans)
                score.append(ans)
                standard.append(1)
                count += 1
                if count % 100 == 0:
                    print(count, 'in 33579 test')
            else:
                pass_count += 1
                if pass_count % 10 == 0:
                    print('pass_count :', pass_count, 'in 33579 test')


    with open('./process_result/neg.txt', 'r') as f:
        print('open neg.txt done')
        count = 0
        pass_count_neg = 0
        for line in f:
            line = line.split()

            bodypart = line[1]
            for index in range(2, len(line)):
                if line[index].isalpha():
                    bodypart += ' '
                    bodypart += line[index]
                else:
                    break
            # bodypart = ' '.join(bodypart)
            if line[0] in vecs.keys() and bodypart in vecs.keys() & geneORGANizer['bodyparts'].keys():
                ans = {cosine_similarity([vecs[line[0]], vecs[bodypart]])[0][1]}
                ans = list(ans)[0]
                score.append(ans)
                standard.append(0)
                count += 1
                if count % 1000 == 0:
                    print(count, 'in 341646 neg')
            else:
                pass_count_neg += 1
                if pass_count_neg % 10 == 0:
                    print('pass_count_neg :', pass_count_neg, 'in 341646 neg')


    # 以上两步，分别是构造正样本和负样本，有关系的对象计算余弦相似度，给予标签1，无关系的对象计算余弦相似度，给予标签0

    f = open('./lzm_mp1_result/' + 'GB170w_sz2' + '_standard_score'+notename+'.pkl', 'wb')
    pickle.dump((standard, score), f)
    f.close()

    f = open('./lzm_mp1_result/' + 'GB170w_sz2' + '_standard'+notename+'.pkl', 'wb')
    pickle.dump(standard, f)
    f.close()
    f = open('./lzm_mp1_result/' + 'GB170w_sz2' + '_score'+notename+'.pkl', 'wb')
    pickle.dump(score, f)
    f.close()


    # f = open('./process_result/' + '5_17_1432' + '_standard_score'+notename+'.pkl', 'rb+')
    # standard, score = pickle.load(f)
    # f.close()


    # #  针对roc_curve的报错
    # standard = np.array(standard)
    # score = np.array(score)


    # g = open('./datamgi/diffpaths/6metapath_add_OGOG/' + 'test_organs' + '_standard_score'+notename+'.pkl', 'rb+')
    # standard_score = pickle.load(g)
    # with open('./datamgi/diffpaths/_standard_score.txt', 'w') as f:
    #     f.write(str(standard_score))

    # (教训)因为没有存standard和score，从.pkl文件中取出来时不熟悉，没法转成roc_curve需要的形式  ：直接存为.pkl，取的时候直接取就行
    # standard_str = standard_score[0]
    # standard_int = []
    # for index in range(len(standard_str)):
    #     standard_int.append(int(standard_str[index]))
    #
    # score_str = standard_score[1]
    # score_flo = []
    # for index in range(len(score_str)):
    #     score_flo.append(float(score_str[index]))

    fpr, tpr, threshold = roc_curve(standard, score)  ###计算真正率和假正率   roc_curve()：第一个参数为标准值，第二个参数为其对应的阳性概率。返回值：fpr假阳性率、tpr真阳性率、threshold是在概率值中从大到小选取，取F1最大时的threshold为该organ的thres
    roc_auc = auc(fpr, tpr)  ### 计算auc的值
    print('roc_auc :', roc_auc, 'threshold : ', threshold)
    with open('./lzm_mp1_result/aucs/GB170w_sz2_roc_auc.txt', 'w') as f:
        f.write('roc_auc:' + str(roc_auc) + 'threshold:' + str(threshold))
    print('AUROC:', getauroc(standard, score))

def lzm_roc_curve_test():
    y = []
    scores = []
    # y = np.array([0, 0, 1, 1])
    # scores = np.array([0.1, 0.4, 0.35, 0.8])
    y.append(0)
    y.append(0)
    y.append(1)
    y.append(1)
    scores.append(0.1)
    scores.append(0.4)
    scores.append(0.35)
    scores.append(0.8)
    fpr, tpr, threshold = roc_curve(y, scores)
    print(fpr)
    print(tpr)
    print(threshold)

def use_model():
    # model = KeyedVectors.load_word2vec_format('./process_result/' + '5_20_0235_full7path_geneORGANizer_metapath_' +'vecs'+ notename +'.txt')
    # model = Word2Vec.load('./process_result/' + '5_20_0235_full7path_geneORGANizer_metapath_' +'vecs'+ notename +'.txt')
    # model = gensim.models.Word2Vec()
    # model = model.wv.save_word2vec_format('./process_result/' + '5_20_0235_full7path_geneORGANizer_metapath_' +'vecs'+ notename +'.txt')
    model = Word2Vec.load('./process_result/' + 'full7pathModel_vec')
    print(model.wv.most_similar(['outer ear'], topn=146))
    print(model.wv.most_similar(['Aars'], topn=50))

def lzm_test_word2vec_save_load():
    common_texts = [['human', 'interface', 'computer'], ['survey', 'user', 'computer', 'system', 'response', 'time'], ['eps', 'user', 'interface', 'system'], ['system', 'human', 'system', 'eps'], ['user', 'response', 'time'], ['trees'], ['graph', 'trees'], ['graph', 'minors', 'trees'], ['graph', 'minors', 'survey']]
    train_model = Word2Vec(common_texts, vector_size=100, window=5, min_count=1, workers=4)

    train_model.save('./MyModel')
    train_model.wv.save_word2vec_format('./mymodel.txt', binary=False)
    train_model.wv.save_word2vec_format('./mymodel1.txt')

    # model = Word2Vec.load('./mymodel.txt')
    # 未解决：存为txt时无法加载
    # model = Word2Vec.wv.save_word2vec_format('./mymodel.txt')
    # model = KeyedVectors.load_word2vec_format('./mymodel.txt')
    # print((model.wv.most_similar(['human'], topn=5)))

def quchong():
    # path = './tutu'
    # if not os.path.exists(path):   
    #     os.makedirs(path)

    f3 = open(f"./quchong.txt","r",encoding='utf-8')
    text_list = []
    chongfu = []
    s = set()
    document = f3.readlines()
    document_num = int(len(document))
    print('原条数：' + str(document_num))
    print('================去重中================')
    content = [x.strip() for x in document]
    # print(content)

    for x in range(0,len(content)):
        url = content[x]
        if url not in s:
            s.add(url)
            text_list.append(url)
        else:
            chongfu.append(url)
    filename = int(len(text_list))
    print('现条数：' + str(filename))
    print('减少了：'+ str(document_num-filename ))


    with open(f'./article/de-duplication.txt','a+',encoding='utf-8') as f:
        for i in range(len(text_list)):
            # s = str(i).split()
            s = str(text_list[i])
            s = s + '\n'
            f.write(s)
        print('================保存文件成功================')
    with open(f'./article/Duplicate items.txt','a+',encoding='utf-8') as f:
        for i in range(len(chongfu)):
            # s = str(i).split()
            s = str(chongfu[i])
            s = s + '\n'
            f.write(s)
        print('================保存文件成功================')



if __name__=='__main__':
    # preprocess()
    # main()
    vecs=importvec('.//lzm_mp1_result/' + 'GB170w_sz2_vec.txt')
    test(vecs)
    # use_model()
    # lzm_roc_curve_test()
    # lzm_test_word2vec_save_load()

    # calthres.cal(Function, goa, go, vecs, path)

    # calthres.lzm_cal1(geneORGANizer, vecs)
    # quchong()

