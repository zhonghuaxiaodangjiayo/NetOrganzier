
import gensim
import importdatas as datas
from random import choice
from gensim.models import Word2Vec, KeyedVectors
from gensim.models.word2vec import LineSentence
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

from Bio.KEGG import REST
from goatools.base import download_go_basic_obo
from goatools.obo_parser import GODag
from goatools.anno.genetogo_reader import Gene2GoReader
import requests
import json
import urllib.request

Function = 'CC'
path = './datahg/'
notename = '-noDis'

# 用于分4个AUC
gO_organs = ['cerebellum','brain','head','uterus','ear','outer ear','middle ear','inner ear','peripheral nerve','peripheral nervous system','cranial nerve','pituitary gland','ovary','testicle','penis','vagina','vulva','skin','hair','prostate','breast','eye','lower limb','spinal cord','mouth','skull','spinal column','vertebrae','pelvis','rib','rib cage','lung','foot','tibia','fibula','femur','sternum','chest wall','clavicle','scapula','humerus','radius','ulna','neck','mandible','maxilla','jaw','ankle','knee','hip','wrist','elbow','shoulder','finger','toe','digit','hand','forearm','arm','shin','thigh','upper limb','face','eyelid','kidney','urethra','heart','anus','urinary bladder','blood vessel','pancreas','bronchus','white blood cell','blood','liver','biliary tract','gallbladder','olfactory bulb','ureter','red blood cell','coagulation system','aorta','heart valve','rectum','tooth','nose','intestine','large intestine','forehead','diaphragm','stomach','small intestine','chin','cheek','bone marrow','lip','pharynx','nail','meninges','cerebrospinal fluid','spleen','lymph node','sinus','abdominal wall','duodenum','esophagus','placenta','larynx','trachea','vocal cord','epiglottis','tongue','scalp','lymph vessel','lacrimal apparatus','adrenal gland','sweat gland','salivary gland','thyroid','hypothalamus','appendix','parathyroid','thymus','fallopian tube','vas deferens']
gO_regions = ['limbs', 'head and neck', 'thorax', 'abdomen', 'pelvis', 'general']
gO_systems = ['immune', 'cardiovascular', 'nervous', 'skeleton', 'skeletal', 'muscle', 'reproductive', 'digestive',	'urinary', 'respiratory', 'endocrine', 'lymphatic', 'integumentary']
gO_germlayers = ['endoderm', 'mesoderm', 'ectoderm']

gene4layer = datas.importgeneORGANizer()


geneORGANizer = datas.importgeneItem()
print('len(bodyparts)', len(geneORGANizer['bodyparts']))
print('len(genes)', len(geneORGANizer['genes']))
print('import geneORGANizer done')

bodypartLinks = datas.importbodyPartLinks()
print('import bodypartLinks done')
# print(bodypartLinks['skull'])
# print(choice(list(bodypartLinks['skull'])))


def preprocess():
    # 导入基因调控网络grn
    global grn
    # 人
    if path[6] == 'h':
        grn = datas.importgrnhg()    #    return {'genes':{gene:re,gene:re...},'res':{re:{'reg':genes,'bind':tfs}},'tfs':{tf:re,tf:re}}
        print('import human grn done')
    # 鼠鼠
    else:
        grn = datas.importgrnmgi()  # return {'genes':{gene:re,gene:re...},'res':{re:{'reg':genes,'bind':tfs}},'tfs':{tf:re,tf:re}}
        print('import mouse grn done')


    # 导入蛋白互作网络PPI
    global ppi
    ppi = datas.importppi(path+'ppifull.txt')
    print('import ppi done')

    print('genes num in geneORGANizer:', len(set(geneORGANizer['genes'])), 'bodyparts num in geneORGANizer:', len(geneORGANizer['bodyparts']))
    print('genes num in grn:', len(set(grn['genes'])))
    print('genes num in ppi:', len(set(ppi.keys())))
    # print('terms num in go:', len(go))

    genetf = set(grn['genes'].keys()) | set(grn['tfs'].keys())


    print('intersection genes of grn_geneORGANizer_ppi num:', len(set(grn['genes'].keys()) & set(geneORGANizer['genes'].keys()) & set(ppi.keys())))
    print('res num:', len(grn['res']))
    print('tfs num:', len(grn['tfs']))


def pickppi(ppisp):
    x = random.randint(0, sum(list(ppisp.values()))-1)
    count = 0
    for item in ppisp.keys():
        count += ppisp[item]
        if count > x:
            return item
    return item

# 从gene开始的游走
def genwalk(walks, genesets, walknum, walklen, geneORGANizer, ppi,logo):
    # 原始函数：def genwalk(walks, genesets, walknum, walklen, goa, go, ppi, logo):
    # 因为有些metapath无法迁移，用不到go、goa、ppi


    # g_O_genes = set(g_O['genes'].keys())
    g_O_organs = set(geneORGANizer['bodyparts'].keys())
    M = len(genesets)
    count = 0
    for gene in genesets:  # 对于每个基因，走了20条拥有50个结点的GBGB...
        for i in range(walknum*8):  # 每个gene节点走20条序列  # 50 ————> 50 *10
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


            # # ...
            # walk = []
            # walk.append(gene)
            # bodypart = choice(list(geneORGANizer['genes'][gene]))
            # walk.append((bodypart))
            # for j in range(walklen):
            #     otherbodypart = choice(list(bodypartLinks[bodypart]))
            #     walk.append((otherbodypart))
            #     bodypart = otherbodypart  # 循环
            # walks.append(walk)

            # 不分层次游走
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

            # 分四个层次进行游走 暂定：每个层次游走60W
            # 胚胎层次
            # walk_germlayer = []
            # walk_germlayer.append(gene)
            # germlayers = choice(list(gene4layer['genesOfGermlayer'][gene]))
            # walk_germlayer.append(germlayers)
            # for j in range(walklen // 2):  # 每条序列走50个节点
            #     gene = choice(list(gene4layer['germlayers'][germlayers]))
            #     walk_germlayer.append(gene)
            #     germlayers = choice(list(gene4layer['genesOfGermlayer'][gene]))
            #     walk_germlayer.append(germlayers)
            # walks.append(walk_germlayer)

            # # 区域层次
            # walk_region = []
            # walk_region.append(gene)
            # regions = choice(list(gene4layer['genesOfRegion'][gene]))
            # walk_region.append(regions)
            # for j in range(walklen // 2):  # 每条序列走50个节点
            #     gene = choice(list(gene4layer['regions'][regions]))
            #     walk_region.append(gene)
            #     regions = choice(list(gene4layer['genesOfRegion'][gene]))
            #     walk_germlayer.append(regions)
            # walks.append(walk_region)
            # # 系统层次
            # walk_system = []
            # walk_system.append(gene)
            # systems = choice(list(gene4layer['genesOfSystem'][gene]))
            # walk_region.append(systems)
            # for j in range(walklen // 2):  # 每条序列走50个节点
            #     gene = choice(list(gene4layer['systems'][systems]))
            #     walk_system.append(gene)
            #     systems = choice(list(gene4layer['genesOfSystem'][gene]))
            #     walk_germlayer.append(systems)
            # walks.append(walk_system)
            # # 组织层次
            # walk_organ = []
            # walk_organ.append(gene)
            # organs = choice(list(gene4layer['genesOfOrgan'][gene]))
            # walk_region.append(organs)
            # for j in range(walklen // 2):  # 每条序列走50个节点
            #     gene = choice(list(gene4layer['organs'][organs]))
            #     walk_organ.append(gene)
            #     systems = choice(list(gene4layer['genesOfOrgans'][gene]))
            #     walk_organ.append(organs)
            # walks.append(walk_organ)

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

def grnwalk(walks, genesets, walknum, walklen, grn, logo):
    M = len(genesets)  # GRTRG
    count = 0
    for gene in genesets:
        for i in range(3):
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
            print(logo, 'No.8P_GRTRT', count, 'grn', M)

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
    # L = len(grn['res'].keys())
    # count = 0
    # for re in grn['res'].keys():
    #     count += 1
    #     if count % 1000 == 0:
    #         print(count, '/res_geneORGANizer', L)
    #     for gene in (grn['res'][re]['reg']) & geneORGANizer['genes'].keys():
    #         for bodypart in geneORGANizer['genes'][gene]:
    #             walks.append([re, bodypart])
    # walks = walks * 40



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


    # tfs = grn['tfs'].keys() & grn['genes'].keys()
    # L = len(tfs)
    # count = 0
    # for tf in tfs:
    #     count += 1
    #     print(count, '/ tfs', L)
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
    #         walks.append(walk)

    # 2022/04/28 in 1:24 返厂改进：没有定义一条从组织开始跑的metapath，且组织之间没有边，导致某些组织没有出现在游走序列中，得不到低维向量表示
    # 2022/05/12 in 10:35 更新至geneORGANizer版本
    # 2022/05/16 in 14:23 更新至geneItem版本
    # 2022/05/19 in 12:45 注释BG看效果
    # count = 0
    # M = len(geneORGANizer['bodyparts'].keys())
    # for bodypart in gO_bodyparts:
    #     # with open(path + 'diffpaths/6metapath_add_OGOG/seq_start_with_organ.txt', 'a+') as f:
    #     #     # f.write('cur organ :', organ) 错误写法
    #     #     f.write('cur organ :' + str(organ) + '\t')
    #     walk = []  # GTGTGTGTGT...——> GBGBGB...——> BGBGBG... 很关键
    #     walk.append(bodypart)
    #
    #     # with open(path + 'diffpaths/6metapath_add_OGOG/seq_start_with_organ.txt', 'a+') as f:
    #     #     f.write('cur organ can choice genes:' + str(list(g_O['organs'][organ])))
    #
    #     gene = choice(list(geneORGANizer['bodyparts'][bodypart]))  # 随机选取一个和当前bodypart有关的基因
    #     # with open(path + 'diffpaths/6metapath_add_OGOG/seq_start_with_organ.txt', 'a+') as f:
    #     #     f.write('organ :' + str(organ) + 'choice gene:' + str(gene))
    #
    #     walk.append(gene)
    #     # 同上一条元路径，先纳入gene和其对应term，
    #     # 对于当前term，将goa网络中，与其对应的gene纳入walk
    #     # 反复操作
    #     for j in range(walklen):                                        # 最终walk：B-G-B-G
    #         bodypart = choice(list(geneORGANizer['genes'][gene]))
    #         walk.append(bodypart)
    #         gene = choice(list(geneORGANizer['bodyparts'][bodypart]))
    #         walk.append(gene)
    #     walks.append(walk)
    #
    #     count += 1
    #     if count % 10 == 0:
    #         print(count, '/bodypart', M)
    #
    # L = len(grn['res'].keys())
    # count = 0
    # for re in grn['res'].keys():        #RGGR RGGR RGGR...iii
    #     count += 1
    #     if count % 100 == 0:
    #         print(count, '/res', L)
    #     walk = [re]
    #     tre = re
    #     for j in range(walklen//4):  #  走满50个节点作为一条序列 该序列中元素都是RGGR RGGR
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

    # # 用多进程跑  #  2022/05/18 16:11 删除gene-bodypart...测试性能变化
    # walksgene = Manager().list()
    # N = 8
    # geneset = []
    # for i in range(N):
    #     geneset.append(set())
    # for i in geneORGANizer['genes'].keys():
    #     geneset[random.randint(0, N - 1)].add(i)
    # processes = []
    # for i in range(N):
    #     processes.append(Process(target=genwalk, args=(walksgene, geneset[i], walknum, walklen, geneORGANizer, 'p'+str(i))))
    # for i in range(N):
    #     processes[i].start()
    # for i in range(N):
    #     processes[i].join()
    # walks = walks+list(walksgene)
    # print(len(walks))


    # M = len(ppi.keys())              #GGGGGGG (PPI) viii
    # count = 0
    # for gene in ppi.keys():
    #     count += 1
    #     for i in range(walknum):
    #         walk = []
    #         walk.append(gene)
    #         for j in range(walklen):
    #             gene = pickppi(ppi[gene])
    #             walk.append(gene)
    #         walks.append(walk)
    #     if count % 100 == 0:
    #         print(count, '/ppi', M)
    #
    #
    # walksgrn = Manager().list()
    # N = 8
    # geneset = []
    # for i in range(N):
    #     geneset.append(set())
    # for i in grn['genes'].keys():
    #     geneset[random.randint(0, N - 1)].add(i)
    # processes = []
    # for i in range(N):         # GRTRG iv
    #     processes.append(
    #         Process(target=grnwalk, args=(walksgrn, geneset[i], walknum, walklen, grn, 'p' + str(i))))
    # for i in range(N):
    #     processes[i].start()
    # for i in range(N):
    #     processes[i].join()
    # walks = walks + list(walksgrn)
    # print(len(walks))
    


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
    w = open('/share/home/liangzhongming/RE_ORGANize_4layers/RE-ORGANizer-master/generateREGOA/LiCO/4Bert/oldpath_5.txt', 'w')
    global walks
    # walks = []

    # 效果不好，从GRN的G出发的路径，借鉴RE-GOA工作：G-R-T-R-T...
    # # 8.G-RE-TF-G...G-B
    # # 从GRN的gene出发，直到走到geneORGANizer中的gene停止
    # # 因为联系需要以交集的3091个gene为媒介，该路径得到GRN中节点与媒介节点的远近关系
    # # 不知道关系远近是否具有传递性？
    # L = len(grn['genes'].keys())
    # count = 0
    # successCount = 0
    # for gene in grn['genes'].keys():
    #     subwalk = []
    #     times = 0
    #     count += 1
    #     if count % 10 == 0:
    #         print(count, '/No.8_G-RE-TF-G...-G-B', L)
    #     subwalk.append(gene)
    #     for i in range(10):
    #         # 使用while + break 来实现do while
    #         while True:
    #             times += 1
    #             re = choice(list(grn['genes'][gene]))
    #             subwalk.append(re)
    #
    #             tf = choice((list(grn['res'][re]['binded'])))
    #             subwalk.append(tf)
    #
    #             gene = choice(list(grn['res'][re]['reg']))
    #             subwalk.append(gene)
    #
    #             if gene in geneORGANizer['genes'].keys() or times >= 10:
    #                 break
    #         if times >= 10:
    #             # print(count, ": times >= 20")
    #             break
    #
    #         bodypart = choice(list(geneORGANizer['genes'][gene]))
    #         subwalk.append(bodypart)
    #         walks.append(subwalk)
    #
    #         successCount += 1
    #         if successCount % 10 == 0:
    #             print(successCount, "success/", L)
    # print(len(walks))

    # # 8P.G-RE-TF-RE-TF
    # walksgrn = Manager().list()
    # N = 4
    # geneset = []
    # for i in range(N):
    #     geneset.append(set())
    # for i in grn['genes'].keys():
    #     geneset[random.randint(0, N - 1)].add(i)
    # processes = []
    # for i in range(N):         # GRTRG
    #     processes.append(
    #         Process(target=grnwalk, args=(walksgrn, geneset[i], walknum, walklen, grn, 'p' + str(i))))
    # for i in range(N):
    #     processes[i].start()
    # for i in range(N):
    #     processes[i].join()
    # walks = walks + list(walksgrn)
    # print(len(walks))

    # M = len(grn['genes'].keys())    # GRTRT...
    # count = 0
    # for gene in grn['genes'].keys():
    #     for i in range(5):
    #         walk = []
    #         walk.append(gene)
    #         for j in range(walklen // 4):
    #             re = choice(list(grn['genes'][gene]))
    #             walk.append(re)
    #             tf = choice(list(grn['res'][re]['binded']))
    #             walk.append(tf)
    #             re = choice(list(grn['tfs'][tf]))
    #             walk.append(re)
    #             tf = choice(list(grn['res'][re]['reg']))  # 师姐写错了 该tf就是Gene
    #             walk.append(tf)
    #         walks.append(walk)
    #     count += 1
    #     if count % 10 == 0:
    #         print('No.8P_GRTRT', count, 'grn', M)



    # 6. TF-G-G-TF
    L = len(grn['tfs'].keys())
    count = 0
    for tf in grn['tfs'].keys():  # 每个tf采样一次，如果该路径走不通，break换下一个tf走，所以最终该元路径会比较少
        # subwalk = []
        count += 1
        if count % 10 == 0:
            print(count, '/No.6_TF-G-G-TF', L)
    
        for i in range(30):
            # walk = []
            # walk.append(tf)
            w.write(str(tf))
            w.write(';')
            for j in range(100 // 3):
                re = choice(list(grn['tfs'][tf]))
                if not list(grn['res'][re]['reg'] & ppi.keys()):
                    break
                gene = choice(list(grn['res'][re]['reg'] & ppi.keys()))
                # walk.append(gene)
                w.write(str(gene))
                w.write(';')
    
                if not list(ppi[gene].keys() & grn['genes'].keys()):
                    break
                tgene = choice(list(ppi[gene].keys() & grn['genes'].keys()))
                # walk.append(tgene)
                w.write(str(tgene))
                w.write(';')
    
                re = choice(list(grn['genes'][tgene]))
                tf = choice((list(grn['res'][re]['binded'])))
                # walk.append(tf)
                w.write(str(tf))
                w.write(';')
            w.write('\n')
        #     subwalk += walk
        # subwalk = subwalk * 40
        # walks.append(subwalk)
    # print('walks length : ', len(walks))
    #
    #
    # 4. TF-RE-G-RE-TF
    L = len(grn['tfs'].keys())
    count = 0
    # subwalk = []
    for tf in grn['tfs'].keys():  # 每个tf采样一次，如果该路径走不通，break换下一个tf走，所以最终该元路径会比较少
        for i in range(75):
            count += 1
            if count % 10 == 0:
                print(count, '/No.4_TF-RE-G-RE-TF', L)
        
            # walk = []
            # walk.append(tf)
            w.write(str(tf))
            w.write(';')
        
            for j in range(100 // 4):
                re = choice(list(grn['tfs'][tf]))
                # walk.append(re)
                w.write(str(re))
                w.write(';')
                '''
                if not list(grn['res'][re]['reg'] & geneORGANizer['genes'].keys()):
                    count_pass_gene += 1
                    print(count_pass_gene)
                    break
                print(list(grn['res'][re]['reg']))  # out:多只有一个基因，因为re调控的基因数量不会太多
                print(list(grn['res'][re]))  # out:['reg', 'binded']
                # 取交集之后很可能变为空集
                gene = choice(list(grn['res'][re]['reg'] & geneORGANizer['genes'].keys()))
                walk.append(gene)
                '''
                if not list(grn['res'][re]['reg'] & ppi.keys()):
                    break
                gene = choice(list(grn['res'][re]['reg'] & ppi.keys()))
                # walk.append(gene)
                w.write(str(gene))
                w.write(';')
        
                re = choice(list(grn['genes'][gene]))
                # walk.append(re)
                w.write(str(re))
                w.write(';')
        
                tf = choice((list(grn['res'][re]['binded'])))
                # walk.append(tf)
                w.write(str(tf))
                w.write(';')
            w.write('\n')
            # subwalk += walk
    
    # print('done pair countOfPath : ', len(subwalk))
    # subwalk = subwalk * 2000   # 500 ————> 1000
    # print(len(subwalk))
    # walks.append(subwalk)
    #
    # 3. G-B-G...
    # 跑这个总是MemoryError
    # 新加入GBBB...和GBBGGBBG
    # walksgene = Manager().list()
    # N = 8
    # geneset = []
    # for i in range(N):
    #     geneset.append(set())
    # for i in geneORGANizer['genes'].keys():
    #     geneset[random.randint(0, N - 1)].add(i)
    # processes = []
    # for i in range(N):
    #     processes.append(Process(target=genwalk, args=(walksgene, geneset[i], walknum, walklen, geneORGANizer, ppi, 'p'+str(i))))
    # for i in range(N):
    #     processes[i].start()
    # for i in range(N):
    #     processes[i].join()
    # walks = walks+list(walksgene)
    # print(len(walks))

    for gene in geneORGANizer['genes'].keys():
        for i in range(50*8):
            # walk = []                                #GTGTGTGTGT...——> GOGOGOGO...
            # walk.append(gene)
            w.write(str(gene))
            w.write(';')
            bodypart = choice(list(geneORGANizer['genes'][gene]))  # 随机选取一个和当前基因有关的组织
            # walk.append(bodypart)
            w.write(str(bodypart))
            w.write(';')
            # 同上一条元路径，先纳入gene和其对应term，
            # 对于当前term，将goa网络中，与其对应的gene纳入walk
            # 反复操作
            for j in range(100 // 2):  # 每条序列走50个节点
                gene = choice(list(geneORGANizer['bodyparts'][bodypart]))
                # walk.append(gene)
                w.write(str(gene))
                w.write(';')
                bodypart = choice(list(geneORGANizer['genes'][gene]))
                # walk.append(bodypart)
                w.write(str(bodypart))
                w.write(';')
            w.write('\n')
            # walks.append(walk)



    #
    #
    # 1. TF-RE-TF-RE...
    tfs = grn['tfs'].keys() & grn['genes'].keys()
    L = len(tfs)
    count = 0
    for tf in tfs:
        # subwalk = []
        count += 1
        print(count, '/ No.1_TF-RE-...', L)
        for i in range(50):
            # walk = [tf]
            w.write(str(tf))
            w.write(';')
            ttf = tf
            for j in range(100 // 2):      # TRTRTRT...(TF-RE-TF-RE...)
                tre = choice(list(grn['genes'][ttf]))
                # walk.append(tre)
                w.write(str(tre))
                w.write(';')
                if list(grn['res'][tre]['binded'] & tfs):
                    ttf = choice(list(grn['res'][tre]['binded'] & tfs))
                    # walk.append(ttf)
                    w.write(str(ttf))
                    w.write(';')
                else:
                    break
            w.write('\n')
    # w.write('\n')
        #     subwalk += walk
        # subwalk = subwalk*10  # 357 * 50(walkNum) * 10 = 178500
        # walks.append(subwalk)
    # with open('./process_result/TF_RE_TF.txt', 'w') as f:
    #     f.write(str(walks))
    # #
    # #
    #
    # 2. RE-G-RE...
    L = len(grn['res'].keys())
    count = 0
    for re in grn['res'].keys():
        count += 1
        if count % 100 == 0:
            print(count, '/No.2_RE-G-RE...', L)
        for i in range(20):
            # walk = []
            # walk.append(re)
            w.write(str(re))
            w.write(';')
            tre = re
            for j in range(50 // 2):
                if not list(grn['res'][tre]['reg']):
                    break
                tgene = choice(list(grn['res'][tre]['reg']))
                # walk.append(tgene)
                w.write(str(tgene))
                w.write(';')
                tre = choice(list(grn['genes'][tgene]))
                # walk.append(tre)
                w.write(str(tre))
                w.write(';')
                # walks.append(walk)
            w.write('\n')
    # w.write('\n')
    print(len(walks))
    # #
    # #
    # 5. RE-G-B-G-RE...
    L = len(grn['res'].keys())
    count = 0
    for re in grn['res'].keys():
        for i in range(50):
            count += 1
            if count % 100 == 0:
                print(count, '/No.5_res_RE-G-B-G-RE', L)
            # walk = [re]
            w.write(str(re))
            w.write(';')
            tre = re
            for j in range(100 // 4):
                if not list(grn['res'][tre]['reg'] & geneORGANizer['genes'].keys()):
                    break
                tgene = choice(list(grn['res'][tre]['reg'] & geneORGANizer['genes'].keys()))
                # walk.append(tgene)
                w.write(str(tgene))
                w.write(';')
        
                bodypart = choice(list(geneORGANizer['genes'][tgene]))
                # walk.append(bodypart)
                w.write(str(bodypart))
                w.write(';')
        
                tgene = choice(list(grn['genes'].keys() & geneORGANizer['bodyparts'][bodypart]))
                # walk.append(tgene)
                w.write(str(tgene))
                w.write(';')
        
                re = choice(list(grn['genes'][tgene]))
                # walk.append(re)
                w.write(str(re))
                w.write(';')
            # walk = walk * 10
            # walks.append(walk)
            w.write('\n')
    print(len(walks))

    #
    # 9. G-G-G...
    M = len(ppi.keys())              #GGGGGGG (PPI) viii
    count = 0
    for gene in ppi.keys():
        count += 1
        for i in range(10):  # //5 ————> //3————> //10
            # walk = []
            # walk.append(gene)
            w.write(str(gene))
            w.write(';')
            for j in range(100 // 2):
                gene = pickppi(ppi[gene])
                # walk.append(gene)
                w.write(str(gene))
                w.write(';')
            # walks.append(walk)
            w.write('\n')
        if count % 100 == 0:
            print(count, '/No.9_G-G-G', M)

    # # 10. G-G-G...G-B
    # # 走到媒介gene结束
    # # 衡量PPI中所有gene和媒介gene的远近关系
    # M = len(ppi.keys())
    # count = 0
    # subwalk = []
    # for gene in ppi.keys():
    #     subwalk.append(gene)
    #     count += 1
    #     for i in range(50):
    #         while True:
    #             gene = pickppi(ppi[gene])
    #             subwalk.append(gene)
    #             if gene in geneORGANizer['genes'].keys():
    #                 break
    #         bodypart = choice(list(geneORGANizer['genes'][gene]))
    #         subwalk.append(bodypart)
    #     if count % 1000 == 0:
    #         print(count, '/No.10_G-G-G...G-B', M)
    #     walks.append(subwalk)
    # print(len(walks))

    # 7.RE-B...
    L = len(grn['res'].keys())
    count_re_b_pair = 0
    count = 0
    for re in grn['res'].keys():
        count += 1
        if count % 1000 == 0:
            print(count, '/NO.7_RE-B...', L)
        for gene in ((grn['res'][re]['reg']) & geneORGANizer['genes'].keys()):
            for bodypart in geneORGANizer['genes'][gene]:
                count_re_b_pair += 1
                for t in range(5):
                    # walks.append([re, bodypart])
                    w.write(str(gene))
                    w.write(';')
                    w.write(str(bodypart))
                    w.write(';')
            w.write('\n')
    print('count_re-b_pair : ', count_re_b_pair)



    # subwalk = []
    # L = len(grn['res'].keys())
    # count_re_b_pair = 0
    # count = 0
    # for re in grn['res'].keys():
    #     count += 1
    #     if count % 1000 == 0:
    #         print(count, '/NO.7_RE-B...', L)
    #     for i in range(20): # 每个re采样次数
    #         gene = choice(list((grn['res'][re]['reg'])))  # 调控的gene
    #         if gene in geneORGANizer['genes'].keys():  # 只考虑与geneORGANizer有交集的部分
    #             subwalk.append(re)
    #             bodypart = choice(list(geneORGANizer['genes'][gene]))
    #             subwalk.append(bodypart)
    #             for j in range(walknum // 2): # 采样路径长度
    #                 gene = choice(list(geneORGANizer['bodyparts'][bodypart]))
    #                 if gene in grn['genes']:
    #                     re = choice(list(grn['genes'][gene]))

    #                 gene = choice(list((grn['res'][re]['reg'])))  # 调控的gene
    #                 if gene in geneORGANizer['genes'].keys():  # 只考虑与geneORGANizer有交集的部分
    #                     subwalk.append(re)
    #                     bodypart = choice(list(geneORGANizer['genes'][gene]))
    #                     subwalk.append(bodypart)

    # walks.append(subwalk)




    # # # L = len(grn['res'].keys())
    # # # count = 0
    # # # for re in grn['res'].keys():  # RGGR RGGR RGGR...iii
    # # #     count += 1
    # # #     if count % 100 == 0:
    # # #         print(count, 'NO.8_RGGR_', L)
    # # #     walk = [re]
    # # #     tre = re
    # # #     for j in range(walklen // 4):  # 走满50个节点作为一条序列 该序列中元素都是RGGR RGGR
    # # #         if not list(grn['res'][tre]['reg'] & ppi.keys()):
    # # #             break
    # # #         tgene = choice(list(grn['res'][tre]['reg'] & ppi.keys()))
    # # #         walk.append(tgene)
    # # #         if not list(ppi[tgene].keys() & grn['genes'].keys()):
    # # #             break
    # # #         tgene = choice(list(ppi[tgene].keys() & grn['genes'].keys()))
    # # #         walk.append(tgene)
    # # #         tre = choice(list(grn['genes'][tgene]))
    # # #         walk.append(tre)
    # # #         tre = choice(list(grn['genes'][tgene]))
    # # #         walk.append(tre)
    # # #     walks.append(walk)
    # # # print(len(walks))
    #
    # # TF-B-TF...
    # L = len(grn['tfs'].keys())
    # count = 0
    # subwalk = []
    # for tf in grn['tfs'].keys():  # 每个tf采样一次，如果该路径走不通，break换下一个tf走，所以最终该元路径会比较少
    #     count += 1
    #     if count % 10 == 0:
    #         print(count, '/No.4_TF-B-TF', L)
    #
    #     walk = []
    #     walk.append(tf)
    #
    #     for j in range(walklen//2):
    #         re = choice(list(grn['tfs'][tf]))
    #
    #         if not list(grn['res'][re]['reg'] & geneORGANizer['genes'].keys()):
    #             break
    #
    #         # gene = choice(list(grn['res'][re]['reg'] & ppi.keys()))
    #         gene = choice(list(grn['res'][re]['reg'] & geneORGANizer['genes'].keys()))
    #
    #         bodypart = choice(list(geneORGANizer['genes'][gene]))
    #         walk.append(bodypart)
    #
    #         tgene = choice(list(grn['genes'].keys() & geneORGANizer['bodyparts'][bodypart]))
    #
    #         re = choice(list(grn['genes'][tgene]))
    #
    #         tf = choice((list(grn['res'][re]['binded'])))
    #         walk.append(tf)
    #
    #     subwalk += walk
    # print('done pair countOfPath : ', len(subwalk))
    # subwalk = subwalk * 500  # 500 ————> 1000
    # print(len(subwalk))
    # walks.append(subwalk)


    # return walks

def lzm_mp2walk_REGOA(walknum, walklen):
    w = open('/share/home/liangzhongming/RE_ORGANize_4layers/RE-ORGANizer-master/generateREGOA/LiCO/4Bert/regoa_path(full).txt', 'w')
    walks = []
    
    M = len(geneORGANizer['genes'])
    count=0
    for gene in geneORGANizer['genes']:
        for i in range(walknum):
            walk=[]                                  # GTTTTTTT...
            walk.append(gene)
            w.write(str(gene))
            w.write(';')
            tissue = choice(list(geneORGANizer['genes'][gene]))
            walk.append(tissue)
            w.write(str(tissue))
            w.write(';')
            for j in range(walklen):
                tissue = choice(list(geneORGANizer['genes'][gene]))
                walk.append(tissue)
                w.write(str(tissue))
                w.write(';')
            w.write('\n')
            walks.append(walk)

            walk = []                                 #GTTGGTTGGTTG...
            walk.append(gene)
            w.write(str(gene))
            w.write(';')
            tissue = choice(list(geneORGANizer['genes'][gene]))
            walk.append(tissue)
            w.write(str(tissue))
            w.write(';')

            ttissue = choice(list(geneORGANizer['genes'][gene]))
            walk.append(ttissue)
            w.write(str(ttissue))
            w.write(';')


            tgene = choice(list(geneORGANizer['bodyparts'][ttissue] & ppi.keys()))
            if tgene is None:
                walks.append(walk)
                w.write('\n')
                continue

            walk.append(tgene)
            w.write(str(tgene))
            w.write(';')

            for j in range(walklen // 2):
                gene = pickppi({x:ppi[tgene][x] for x in ppi[tgene].keys() if x in geneORGANizer['genes']})
                walk.append(gene)
                w.write(str(gene))
                w.write(';')
                tissue = choice(list(geneORGANizer['genes'][gene]))
                walk.append(tissue)
                w.write(str(tissue))
                w.write(';')

                ttissue = choice(list(geneORGANizer['genes'][gene]))
                walk.append(ttissue)
                w.write(str(ttissue))
                w.write(';')

                tgene = choice(list(geneORGANizer['bodyparts'][ttissue] & ppi.keys()))
                if tgene is None:
                    walks.append(walk)
                    w.write('\n')
                    break

                walk.append(tgene)
                w.write(str(tgene))
                w.write(';')

                #print(len(walk), walk)
            w.write('\n')
            walks.append(walk)


        for j in range(50*8):
            walk = []                                #GTGTGTGTGT...
            walk.append(gene)
            w.write(str(gene))
            w.write(';')
            tissue = choice(list(geneORGANizer['genes'][gene]))
            walk.append(tissue)
            w.write(str(tissue))
            w.write(';')
            for j in range(walklen):
                gene = choice(list(geneORGANizer['bodyparts'][tissue]))
                walk.append(gene)
                w.write(str(gene))
                w.write(';')
                tissue = choice(list(geneORGANizer['genes'][gene]))
                walk.append(tissue)
                w.write(str(tissue))
                w.write(';')
            w.write('\n')
            walks.append(walk)
        
        count += 1
        if count % 10 == 0:
            print(count, 'gene', M)

    M = len(grn['genes'])                               #GRTRG
    count = 0
    for gene in grn['genes']:
        for i in range(walknum * 5):
            walk = []
            walk.append(gene)
            w.write(str(gene))
            w.write(';')
            for j in range(walklen // 4):
                re = choice(list(grn['genes'][gene]))
                walk.append(re)
                w.write(str(re))
                w.write(';')
                tf = choice(list(grn['res'][re]['binded']))
                walk.append(tf)
                w.write(str(tf))
                w.write(';')
                re = choice(list(grn['tfs'][tf]))
                walk.append(re)
                w.write(str(re))
                w.write(';')
                tf = choice(list(grn['res'][re]['reg']))
                walk.append(tf)
                w.write(str(tf))
                w.write(';')
            w.write('\n')
            walks.append(walk)
        count += 1
        if count % 100 == 0:
            print(count, 'grn', M)


    L = len(grn['res'].keys())      # RE-B
    count=0
    for re in grn['res']:
        count+=1
        if count%100 == 0:
            print(count,'/re-tissue',L)
        walk = []
        for gene in (grn['res'][re]['reg']):
            if gene in geneORGANizer['genes']:
                for i in range(5):
                    for tissue in geneORGANizer['genes'][gene]:
                        walk.append(re)
                        walk.append(tissue)
                        w.write(str(re))
                        w.write(';')
                        w.write(str(tissue))
                        w.write(';')
            else:
                continue
        if walk != []:
            w.write('\n')
            walks.append(walk)



    L = len(grn['res'].keys())
    count=0
    for re in grn['res'].keys():        #RGGRRGGRRGGR...
        count+=1
        if count%100 == 0:
            print(count,'/RGGR',L)
        walk = [re]
        w.write(str(re))
        w.write(';')
        tre = re
        for j in range(walklen//4):
            if not list(grn['res'][tre]['reg'] & ppi.keys()):
                break
            tgene = choice(list(grn['res'][tre]['reg'] & ppi.keys()))
            walk.append(tgene)
            w.write(str(tgene))
            w.write(';')
            if not list(ppi[tgene].keys() & grn['genes'].keys()):
                break
            tgene = choice(list(ppi[tgene].keys() & grn['genes'].keys()))
            walk.append(tgene)
            w.write(str(tgene))
            w.write(';')
            tre = choice(list(grn['genes'][tgene]))
            walk.append(tre)
            w.write(str(tre))
            w.write(';')
            tre = choice(list(grn['genes'][tgene]))
            walk.append(tre)
            w.write(str(tre))
            w.write(';')
        w.write('\n')
        walks.append(walk)
    print(len(walks))



    M = len(ppi.keys())              #GGGGGGG (PPI)
    count=0
    for gene in ppi.keys():
        count+=1
        for i in range(walknum // 2):
            walk=[]
            walk.append(gene)
            w.write(str(gene))
            w.write(';')
            for j in range(walklen):
                gene=pickppi(ppi[gene])
                walk.append(gene)
                w.write(str(gene))
                w.write(';')
            w.write('\n')
            walks.append(walk)
        if count%100==0:
            print(count,'/ppi',M)
    
    return walks




def path4Bert(walknum, walklen):
    '''
    处理成类似语句，非对称路径
    1.TF-RE-G-B
    2.TF-RE-G-G-B

    缺点：缺少长程的关系，如同类型节点间的关系可能无法学习好。需要补充长程语句
    3.TF-RE-G-B-G-RE-TF...
    4.TF-RE-G-G-B-G-G-RE-TF...

    5.G-B-B... VS G-B-G...
    '''
    w = open('/share/home/liangzhongming/RE_ORGANize_4layers/RE-ORGANizer-master/generateREGOA/LiCO/4Bert/newPath.txt', 'w')

    '''1.TF-RE-G-B-G-RE-TF '''
    cnt_0 = 0
    for tfA in grn['tfs']: # 636
        for i in range(walknum): # 20
            reA = choice(list(grn['tfs'][tfA])) # 1

            j = 0
            geneA = choice(list(grn['res'][reA]['reg']))
            while geneA not in geneORGANizer['genes']:
                j += 1
                geneA = choice(list(grn['res'][reA]['reg']))
                if j == 10:
                    break
            if geneA not in geneORGANizer['genes']: # 没有合适的gene 放弃这条路径
                break

            for bodypart in geneORGANizer['genes'][geneA]: # n2

                j = 0
                geneB = choice(list(geneORGANizer['bodyparts'][bodypart]))
                while geneB not in grn['genes']:
                    j += 1
                    geneB = choice(list(geneORGANizer['bodyparts'][bodypart]))
                    if j == 10:
                        break
                if geneB not in grn['genes']: # 没有合适的gene 放弃这条路径
                    break

                reB = choice(list(grn['genes'][geneB])) # 1
                tfB = choice(list(grn['res'][reB]['binded']))

                w.write(str(tfA))
                w.write(';')
                w.write(str(reA))
                w.write(';')
                w.write(str(geneA))
                w.write(';')
                w.write(str(bodypart))
                w.write(';')

                w.write(str(geneB))
                w.write(';')
                w.write(str(reB))
                w.write(';')
                w.write(str(tfB))
                w.write(';')

                w.write('\n')
                cnt_0 += 1
                if cnt_0 % 1000 == 0:
                    print(cnt_0, '/No.1_TF-RE-G-B-G-RE-TF')


    cnt_1 = 0
    '''2.RE-G-B-G-RE '''
    for reA in grn['res']:
        for i in range(5): # 14w个RE 每个走5次

            j = 0
            geneA = choice(list(grn['res'][reA]['reg']))
            while geneA not in geneORGANizer['genes']:
                j += 1
                geneA = choice(list(grn['res'][reA]['reg']))
                if j == 10:
                    break
            if geneA not in geneORGANizer['genes']: # 没有合适的gene 放弃这条路径
                break

            for bodypart in geneORGANizer['genes'][geneA]:

                j = 0
                geneB = choice(list(geneORGANizer['bodyparts'][bodypart]))
                while geneB not in grn['genes']:
                    j += 1
                    geneB = choice(list(geneORGANizer['bodyparts'][bodypart]))
                    if j == 10:
                        break
                if geneB not in grn['genes']: # 没有合适的gene 放弃这条路径
                    break

                reB = choice(list(grn['genes'][geneB]))

                w.write(str(reA))
                w.write(';')
                w.write(str(geneA))
                w.write(';')
                w.write(str(bodypart))
                w.write(';')
                w.write(str(geneB))
                w.write(';')
                w.write(str(reB))
                w.write(';')

                w.write('\n')
                cnt_1 += 1
                if cnt_1 % 1000 == 0:
                    print(cnt_1, '/No.2_RE-G-B-G-RE')




    # cnt_2 = 0
    # '''3.G-G-B-G-G '''
    # for geneA in ppi.keys():

    #     j = 0
    #     geneB = pickppi(ppi[geneA])
    #     while geneB not in geneORGANizer['genes'].keys():
    #         j += 1
    #         geneB = pickppi(ppi[geneA])
    #         if j == 10:
    #             break
    #     if geneB not in geneORGANizer['genes'].keys(): # 没有合适的gene 放弃这条路径
    #         break
        
    #     for bodypart in geneORGANizer['genes'][geneB]:
    #         geneC = choice(list(geneORGANizer['bodyparts'][bodypart]))
    #         while geneC not in ppi.keys():
    #             j += 1
    #             geneC = choice(list(geneORGANizer['bodyparts'][bodypart]))
    #         if j == 10:
    #             break
    #         if geneC not in ppi.keys():
    #             break

    #         geneD = pickppi(ppi[geneC])

    #         w.write(str(geneA))
    #         w.write(';')
    #         w.write(str(geneB))
    #         w.write(';')
    #         w.write(str(bodypart))
    #         w.write(';')
    #         w.write(str(geneC))
    #         w.write(';')
    #         w.write(str(geneD))
    #         w.write(';')

    #         w.write('\n')
    #         cnt_2 += 1
    #         if cnt_2 % 1000 == 0:
    #             print(cnt_2, '/No.3_G-G-B-G-G')


    # cnt_3 = 0
    # '''4.G-B-G-B-G '''
    # for geneA in geneORGANizer['genes'].keys(): # 3490
    #     for bodypartA in geneORGANizer['genes'][geneA]: # 146
    #         geneB = choice(list(geneORGANizer['bodyparts'][bodypartA]))

    #         for bodypartB in geneORGANizer['genes'][geneB]:
    #             geneC = choice(list(geneORGANizer['bodyparts'][bodypartB]))

    #             w.write(str(geneA))
    #             w.write(';')
    #             w.write(str(bodypartA))
    #             w.write(';')
    #             w.write(str(geneB))
    #             w.write(';')
    #             w.write(str(bodypartB))
    #             w.write(';')
    #             w.write(str(geneC))
    #             w.write(';')

    #             w.write('\n')
    #             cnt_3 += 1
    #             if cnt_3 % 1000 == 0:
    #                 print(cnt_3, '/No.4_G-B-G-B-G')

def main():


    # walknum=20  # 从每个节点开始游走出walknum条路径
    # walklen=50  # 每条路径50个节点
    # walks=mp2vwalk(walknum, walklen)

    # walknum=50  # 从每个节点开始游走出walknum条路径
    # walklen=100  # 每条路径walklen个节点
    # walks=lzm_mp2walk(walknum, walklen)

    # walknum=20  # 从每个节点开始游走出walknum条路径
    # walklen=50  # 每条路径walklen个节点
    # walks=lzm_mp2walk_REGOA(walknum, walklen)


    # walknum=50  # 从每个节点开始游走出walknum条路径
    # walklen=100  # 每条路径walklen个节点
    # walks=path4Bert(walknum, walklen)

    # f = open('./LiCO/walks/RE_B_20_100' + '_walks' + '.txt', 'w')
    # f.write(str(walks))
    # f.close()


    # #
    # # global model
    # #
    #
    # f = open('./LiCO/walks/RE_B_20_100' + '_walks' + '.pkl', 'wb')
    # pickle.dump(walks, f)
    # f.close()
    #
    # f = open('./LiCO/walks/RE_B_20_100' + '_walks' + '.txt', 'w')
    # f.write(str(walks))
    # f.close()


    # w = open('/share/home/liangzhongming/RE_ORGANize_4layers/RE-ORGANizer-master/generateREGOA/LiCO/4Bert/test.txt', 'w')
    # g = open('/share/home/liangzhongming/RE_ORGANize_4layers/RE-ORGANizer-master/generateREGOA/LiCO/4Bert/regoa_path(full).pkl', 'rb+')
    # walks = pickle.load(g)
    # print(len(walks)) # 2551504
    # walks = [sent for sent in walks if sent != []]
    # print(len(walks)) # 2463659


    # sentences = LineSentence('/share/home/liangzhongming/RE_ORGANize_4layers/RE-ORGANizer-master/generateREGOA/LiCO/4Bert/regoa_path_full.txt')
    # f = open('/share/home/liangzhongming/RE_ORGANize_4layers/RE-ORGANizer-master/generateREGOA/LiCO/4Bert/regoa_path(full).pkl', 'wb')
    # pickle.dump(walks, f)
    # f.close()

    # f = open('/share/home/liangzhongming/RE_ORGANize_4layers/RE-ORGANizer-master/generateREGOA/LiCO/4Bert/regoa_path(full)_list.txt', 'w')
    # f.write(str(walks))
    # f.close()

    # g = open('/share/home/liangzhongming/RE_ORGANize_4layers/RE-ORGANizer-master/generateREGOA/LiCO/4Bert/regoa_path(full).pkl', 'rb+')
    # walks = pickle.load(g)
    # print(len(walks))

    g = open('/share/home/liangzhongming/RE_ORGANize_4layers/RE-ORGANizer-master/generateREGOA/model0.87/path/onlyGB_140w_4layerTrain_metapath_walks-noDis.pkl', 'rb+')
    walks = pickle.load(g)
    print(len(walks))

    # g = open('/share/home/liangzhongming/RE_ORGANize_4layers/RE-ORGANizer-master/generateREGOA/LiCO/4Bert/regoa_path(full).txt', 'r')
    # walks = pickle.load(g)
    # print(len(walks))

    # vocab = [tuple(map(str, sentence)) for sentence in walks]
    # f = open('/share/home/liangzhongming/RE_ORGANize_4layers/RE-ORGANizer-master/generateREGOA/LiCO/4Bert/regoa_path(full)_tuple.txt', 'w')
    # f.write(str(vocab))
    # f.close()

    # g = open('/share/home/liangzhongming/RE_ORGANize_4layers/RE-ORGANizer-master/generateREGOA/LiCO/4Bert/regoa_path(full).txt', 'r')
    # walks = pickle.load(g)
    # print(len(walks))


    print('walk done')

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

    # g = open('./model0.87/path/' + '10_12_6path_noGB_4layerTrain_humanGRN_metapath' + '_walks' + notename + '.pkl', 'rb+')
    # walks = pickle.load(g)
    #
    # print('walk done')


    # model = Word2Vec(walks, vector_size=200, window=2, min_count=0, sg=1, workers=16, epochs=10)
    # print('train done')

    # model.save('/share/home/liangzhongming/RE_ORGANize_4layers/RE-ORGANizer-master/generateREGOA/LiCO/4Bert/regoaPath_sz2_vec')
    # model.wv.save_word2vec_format('/share/home/liangzhongming/RE_ORGANize_4layers/RE-ORGANizer-master/generateREGOA/LiCO/4Bert/regoaPath_sz2_vec.txt')


    # # # 再训练
    model = gensim.models.Word2Vec.load('/share/home/liangzhongming/RE_ORGANize_4layers/RE-ORGANizer-master/generateREGOA/LiCO/4Bert/3_27_regoaPath_4layerTrain_humanGRN_vec')
    #
    # # g = open('./model0.87/path/' + 'onlyGB_140w_4layerTrain_metapath' + '_walks' + notename + '.pkl', 'rb+')
    # # walks = pickle.load(g)
    # print('walk done')
    #
    model.build_vocab(walks, update=True)
    model.train(walks, total_examples=model.corpus_count, epochs=10)
    print('train done')
    #
    model.save('/share/home/liangzhongming/RE_ORGANize_4layers/RE-ORGANizer-master/generateREGOA/LiCO/4Bert/regoa_path(full)_addGB_vec')
    model.wv.save_word2vec_format('/share/home/liangzhongming/RE_ORGANize_4layers/RE-ORGANizer-master/generateREGOA/LiCO/4Bert/regoa_path(full)_addGB_vec.txt')

    
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
    plt.savefig("./article/fig/regoa_path(full)_addGB" + ".jpg")
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

    score_germlayers = []
    standard_germlayers = []

    score_regions = []
    standard_regions = []

    score_systems = []
    standard_systems = []

    score_organs = []
    standard_organs = []

    standard = []  # 空字典是{}  空集合set()  空列表[]或list()
    score = []
    # './model0.87/data/test_total4Layertest.txt'
    with open('./model0.87/data/DX_test_total4Layer.txt', 'r') as f:
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
            # 拿到bodypart后，区分其属于geneORGANizer的哪个层次，append入不同的score
            if line[0] in vecs.keys() and bodypart in vecs.keys() & geneORGANizer['bodyparts'].keys():  # 如果gene存在vecs文件中，且bodypart存在vecs文件和geneORGANizer文件中
                # 统一一个AUC
                ans = {cosine_similarity([vecs[line[0]], vecs[bodypart]])[0][1]}
                ans = list(ans)[0]
                score.append(ans)
                standard.append(1)

                if bodypart in gO_germlayers:
                    ans_germlayers = {cosine_similarity([vecs[line[0]], vecs[bodypart]])[0][1]}
                    ans_germlayers = list(ans_germlayers)[0]
                    score_germlayers.append(ans_germlayers)
                    standard_germlayers.append(1)

                if bodypart in gO_regions:
                    ans_regions = {cosine_similarity([vecs[line[0]], vecs[bodypart]])[0][1]}
                    ans_regions = list(ans_regions)[0]
                    score_regions.append(ans_regions)
                    standard_regions.append(1)

                if bodypart in gO_systems:
                    ans_systems = {cosine_similarity([vecs[line[0]], vecs[bodypart]])[0][1]}
                    ans_systems = list(ans_systems)[0]
                    score_systems.append(ans_systems)
                    standard_systems.append(1)

                if bodypart in gO_organs:
                    ans_organs = {cosine_similarity([vecs[line[0]], vecs[bodypart]])[0][1]}
                    ans_organs = list(ans_organs)[0]
                    score_organs.append(ans_organs)
                    standard_organs.append(1)

                count += 1
                if count % 100 == 0:
                    print(count, 'in 33579 test')
            else:
                pass_count += 1
                if pass_count % 10 == 0:
                    print('pass_count :', pass_count, 'in 33579 test')


    with open('./model0.87/data/DX_neg.txt', 'r') as f:
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
                # # 统一计算AUC
                ans = {cosine_similarity([vecs[line[0]], vecs[bodypart]])[0][1]}
                ans = list(ans)[0]
                score.append(ans)
                standard.append(0)

                if bodypart in gO_germlayers:
                    ans_germlayers = {cosine_similarity([vecs[line[0]], vecs[bodypart]])[0][1]}
                    ans_germlayers = list(ans_germlayers)[0]
                    score_germlayers.append(ans_germlayers)
                    standard_germlayers.append(0)

                if bodypart in gO_regions:
                    ans_regions = {cosine_similarity([vecs[line[0]], vecs[bodypart]])[0][1]}
                    ans_regions = list(ans_regions)[0]
                    score_regions.append(ans_regions)
                    standard_regions.append(0)

                if bodypart in gO_systems:
                    ans_systems = {cosine_similarity([vecs[line[0]], vecs[bodypart]])[0][1]}
                    ans_systems = list(ans_systems)[0]
                    score_systems.append(ans_systems)
                    standard_systems.append(0)

                if bodypart in gO_organs:
                    ans_organs = {cosine_similarity([vecs[line[0]], vecs[bodypart]])[0][1]}
                    ans_organs = list(ans_organs)[0]
                    score_organs.append(ans_organs)
                    standard_organs.append(0)

                count += 1
                if count % 1000 == 0:
                    print(count, 'in 341646 neg')
            else:
                pass_count_neg += 1
                if pass_count_neg % 10 == 0:
                    print('pass_count_neg :', pass_count_neg, 'in 341646 neg')


    # 以上两步，分别是构造正样本和负样本，有关系的对象计算余弦相似度，给予标签1，无关系的对象计算余弦相似度，给予标签0

    # f = open('./lzm_mp2_result/' + '7_4_only_GBB_GBBG_170wnode_sz8' + '_standard_score'+notename+'.pkl', 'wb')
    # pickle.dump((standard, score), f)
    # f.close()
    #
    # f = open('./lzm_mp2_result/' + '7_4_only_GBB_GBBG_170wnode_sz8' + '_standard'+notename+'.pkl', 'wb')
    # pickle.dump(standard, f)
    # f.close()
    # f = open('./lzm_mp2_result/' + '7_4_only_GBB_GBBG_170wnode_sz8' + '_score'+notename+'.pkl', 'wb')
    # pickle.dump(score, f)
    # f.close()


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

    # 分为四个层次计算AUC
    # 胚层
    fpr_germlayers, tpr_germlayers, threshold_germlayers = roc_curve(standard_germlayers, score_germlayers)
    roc_auc_germlayers = auc(fpr_germlayers, tpr_germlayers)
    with open('./article/fig/GL_regoa_path(full)_addGB.txt', 'w') as f:
        f.write('roc_auc_germlayers:' + str(roc_auc_germlayers) + 'threshold:' + str(threshold_germlayers))
    print('AUROC:', getauroc(standard_germlayers, score_germlayers))

    # 区域
    fpr_regions, tpr_regions, threshold_regions = roc_curve(standard_regions, score_regions)
    roc_auc_regions = auc(fpr_regions, tpr_regions)

    with open('./article/fig/GR_regoa_path(full)_addGB.txt', 'w') as f:
        f.write('roc_auc_regions:' + str(roc_auc_regions) + 'threshold:' + str(threshold_regions))
    print('AUROC:', getauroc(standard_regions, score_regions))

    # 系统
    fpr_systems, tpr_systems, threshold_systems = roc_curve(standard_systems, score_systems)
    roc_auc_systems = auc(fpr_systems, tpr_systems)

    with open('./article/fig/GS_regoa_path(full)_addGB.txt', 'w') as f:
        f.write('roc_auc_systems:' + str(roc_auc_systems) + 'threshold:' + str(threshold_systems))
    print('AUROC:', getauroc(standard_systems, score_systems))

    # 组织
    fpr_organs, tpr_organs, threshold_organs = roc_curve(standard_organs, score_organs)
    roc_auc_organs = auc(fpr_organs, tpr_organs)

    with open('./article/fig/GO_regoa_path(full)_addGB.txt', 'w') as f:
        f.write('roc_auc_organs:' + str(roc_auc_organs) + 'threshold:' + str(threshold_organs))
    print('AUROC:', getauroc(standard_organs, score_organs))


    # # 统一计算AUC
    fpr, tpr, threshold = roc_curve(standard, score)  ###计算真正率和假正率   roc_curve()：第一个参数为标准值，第二个参数为其对应的阳性概率。返回值：fpr假阳性率、tpr真阳性率、threshold是在概率值中从大到小选取，取F1最大时的threshold为该organ的thres
    roc_auc = auc(fpr, tpr)  ### 计算auc的值
    print('roc_auc :', roc_auc, 'threshold : ', threshold)
    with open('./article/fig/total_regoa_path(full)_addGB.txt', 'w') as f:
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


def build_vocab_of_bert():
    cnt_re = 0
    cnt_tf = 0
    cnt_gene = 0
    nodes = set()
    with open('./vocab,txt', 'w') as fw:
        # grn
        for re in grn['res'].keys():
            nodes.add(re)
        for gene in grn['genes'].keys():
            nodes.add(gene)
        for tf in grn['tfs'].keys():
            nodes.add(tf)
        # ppi
        for protein in ppi.keys():
            nodes.add(protein)

        # geneOrganizer
        for gene in geneORGANizer['genes'].keys():
            nodes.add(gene)
        for bodypart in geneORGANizer['bodyparts'].keys():
            nodes.add(bodypart)

        print(len(nodes))

        cnt = 0
        for i in range(0, len(nodes)):
            fw.write(nodes[i])
            fw.write('\n')
            cnt += 1
        print('cnt_write : ', cnt)


def pathway():
    # 获取KEGG中人类通路列表
    pathways = REST.kegg_list("pathway", "hsa").read().rstrip().split("\n")

    # proxy_support = urllib.request.ProxyHandler({"http":"http://127.0.0.1:7890"})
    # opener = urllib.request.build_opener(proxy_support)
    # urllib.request.install_opener(opener)

    # pathways = urllib.request.urlopen("http://rest.kegg.jp/list/pathway/hsa").read().decode("utf-8").strip().split("\n")
    pathway_ids = [p.split("\t")[0] for p in pathways]

    # 将每个通路及其基因集合写入本地txt文件
    with open('/share/home/liangzhongming/human_pathways.txt', 'w') as f:
        for pid in pathway_ids:
            # 获取通路名称和基因集合
            pathway_file = REST.kegg_get(pid).read()
            name_line = [l for l in pathway_file.rstrip().split("\n") if l.startswith("NAME")][0]
            pathway_name = name_line.split("  ")[1]
            gene_set = set()
            for line in pathway_file.rstrip().split("\n"):
                if line.startswith(" "):
                    continue
                fields = line.rstrip().split("\t")
                if fields[0].startswith("hsa:"):
                    gene_id = fields[0].split(":")[1]
                    gene_set.add(gene_id)
            gene_set_str = ", ".join(sorted(gene_set))

            # 写入txt文件
            f.write(f"{pathway_name}\t{gene_set_str}\n")



def goterm():
    # 从本地OBO文件加载GO term
    # obo_file = "go-basic.obo"
    obo_file = "/share/home/liangzhongming/go-basic.obo"
    godag = GODag(obo_file)

    # 从本地文件加载基因- GO term注释
    gene2go_file = "/share/home/liangzhongming/gene2go"
    gene2go_reader = Gene2GoReader(gene2go_file, godag)

    # 获取人类的GO term
    human_terms = set()
    for gene_id, term_list in gene2go_reader.get_items("9606"):
        print(f"len(term_list):{len(term_list)}")
        for term in term_list:
            human_terms.add(term)

    # 将每个term及其基因集合写入本地txt文件
    with open("/share/home/liangzhongming/human_go_terms.txt", "w") as f:
        print(f"write into local file")
        for term in human_terms:
            gene_set = set(gene2go_reader.get_ns2assc()[term])
            gene_set_str = ", ".join(sorted(gene_set))
            f.write(f"{term}\t{gene_set_str}\n")


def celltype():
    # 发送GET请求获取所有人类细胞类型信息
    response = requests.get("http://biocc.hrbmu.edu.cn/CellMarker/api/get_cell_type?species=human")
    cell_types = json.loads(response.text)

    # 将每个细胞类型及其基因集合写入本地txt文件
    with open("/share/home/liangzhongming/human_cell_types.txt", "w") as f:
        for cell_type in cell_types:
            gene_set = set(cell_type["gene"].split(","))
            gene_set_str = ", ".join(sorted(gene_set))
            f.write(f"{cell_type['cell_type']}\t{gene_set_str}\n")


if __name__=='__main__':
    # preprocess()
    # main()
    # vecs=importvec('./model0.87/' + '10_12_7path_addGB_4layerTrain_humanGRNh_vec.txt')
    # vecs=importvec('./article/model/miniPath2Test_embeddings.txt')
    # vecs=importvec('./article/model/miniPath2Train_embeddings.txt')
    # vecs=importvec('/share/home/liangzhongming/RE_ORGANize_4layers/RE-ORGANizer-master/generateREGOA/LiCO/4Bert/regoa_path(full)_addGB_vec.txt')
    # test(vecs)
    # use_model()
    # lzm_roc_curve_test()
    # lzm_test_word2vec_save_load()

    # calthres.cal(Function, goa, go, vecs, path)

    # calthres.lzm_cal1(geneORGANizer, vecs)

    # build_vocab_of_bert()

    # lzm_mp2walk(50, 100)

    # path4Bert(20, 100)

    # pathway()
    # celltype()
    goterm()