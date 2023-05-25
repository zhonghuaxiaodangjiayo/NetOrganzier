import importdatas as datas
from sklearn.metrics.pairwise import cosine_similarity
from multiprocessing import Process, Queue, Manager
import random
import pickle
import heapq
from collections import OrderedDict
import numpy as np
import dictdiffer
from sklearn.metrics import precision_recall_curve

from sklearn import preprocessing

import matplotlib
import matplotlib.pyplot as plt

from scipy import stats
import scipy.stats
from scipy.stats import levene

# 糖尿病相关组织
diabetes_organs = ['abdominal wall', 'ankle', 'arm', 'blood', 'blood vessel', 'brain', 'chest wall', 'clavicle',
                   'digit', 'ear', 'elbow', 'eye', 'eyelid', 'face', 'femur', 'fibula', 'finger', 'foot', 'forearm',
                   'forehead', 'hand', 'heart', 'hip', 'jaw', 'kidney', 'knee', 'lip', 'liver', 'mandible', 'maxilla',
                   'mouth', 'neck', 'nose', 'ovary', 'pancreas', 'penis', 'peripheral nerve', 'placenta', 'radius',
                   'rib', 'rib cage', 'scapula', 'shin', 'shoulder', 'skin', 'skull', 'spinal column', 'sternum',
                   'thigh', 'tibia', 'toe', 'ulna', 'uterus', 'vertebrae', 'white blood cell', 'wrist']


# geneORGANizer中的组织
gO_organs = ['cerebellum','brain','head','uterus','ear','outer ear','middle ear','inner ear','peripheral nerve','peripheral nervous system','cranial nerve','pituitary gland','ovary','testicle','penis','vagina','vulva','skin','hair','prostate','breast','eye','lower limb','spinal cord','mouth','skull','spinal column','vertebrae','pelvis','rib','rib cage','lung','foot','tibia','fibula','femur','sternum','chest wall','clavicle','scapula','humerus','radius','ulna','neck','mandible','maxilla','jaw','ankle','knee','hip','wrist','elbow','shoulder','finger','toe','digit','hand','forearm','arm','shin','thigh','upper limb','face','eyelid','kidney','urethra','heart','anus','urinary bladder','blood vessel','pancreas','bronchus','white blood cell','blood','liver','biliary tract','gallbladder','olfactory bulb','ureter','red blood cell','coagulation system','aorta','heart valve','rectum','tooth','nose','intestine','large intestine','forehead','diaphragm','stomach','small intestine','chin','cheek','bone marrow','lip','pharynx','nail','meninges','cerebrospinal fluid','spleen','lymph node','sinus','abdominal wall','duodenum','esophagus','placenta','larynx','trachea','vocal cord','epiglottis','tongue','scalp','lymph vessel','lacrimal apparatus','adrenal gland','sweat gland','salivary gland','thyroid','hypothalamus','appendix','parathyroid','thymus','fallopian tube','vas deferens']

GTEx_organs = ['uterus', 'blood vessel', 'brain', 'large intestine', 'heart', 'kidney', 'skin', 'adrenal gland', 'urinary bladder', 'breast', 'fallopian tube', 'liver', 'lung', 'salivary gland', 'skeletal muscle', 'shin', 'ovary', 'pancreas', 'pituitary gland', 'prostate', 'small intestine', 'spleen', 'stomach', 'thyroid', 'vagina', 'blood', 'testicle']

def getthres():

    fmax = dict()

    bodyparts = set()
    with open('./process_result/' + '5_20_0235_full7path_organs_threshold.txt', 'r') as f:  # 由文件'calthres.py'中的方法cal()得到，其中是term和对应的threshold
        next(f)  # 第一行是标题
        for line in f:
            line = line.split()  # 格式：gene  threshigh  thresfmax  threslow
            # 因为组织名字有可能为 xxx xxx xxx 故需要同importvecs()一样处理
            obj = line[0]
            for index in range(1, len(line)):  # organs名字可能是'xxx xxx xxx'形式
                if line[index].isalpha():
                    obj += ' '
                    obj += line[index]
                else:
                    break
            fmax[obj] = float(line[index + 1])  # 得到[term——thres]

            bodyparts.add(obj)  # 得到一系列具有阈值threshold的term
    return fmax, bodyparts

def generateDict_gene_predict_bodypart():
    vecs = importvec('./LiCO/' + '10_15_7path_addGB_4layerTrain_humanGRN_2_vec.txt')
    geneORGANizer = datas.importgeneItem()
    grn = datas.importgrnhg()
    ppi = datas.importppi()

    prob = dict()
    gene_predict_bodypart_dict = dict()

    # geneORGANizer gene

    M = len(ppi.keys())
    count = 0
    for gene in geneORGANizer['genes'].keys():
        if gene in vecs:
            if gene not in gene_predict_bodypart_dict:
                gene_predict_bodypart_dict[gene] = set()
            prob[gene] = OrderedDict()
            for bodypart in geneORGANizer['bodyparts'].keys():
                prob[gene][bodypart] = cosine_similarity([vecs[gene], vecs[bodypart]])[0][1]

            vd = OrderedDict(sorted(prob[gene].items(), key=lambda t: t[1], reverse=True))

            i = 0
            for k in vd.keys():
                gene_predict_bodypart_dict[gene].add(k)
                i += 1
                if i == 11:
                    break

            count = count + 1
            if count % 10 == 0:
                print(count, ' / ', M)
    return gene_predict_bodypart_dict

def generateDict_bodypart_predict_gene():
    # RNA_Atlas_organs = ['adipose', 'large intestine', 'heart', 'hypothalamus', 'kidney', 'liver', 'lung', 'ovary',
    #                     'skeletal muscle', 'spleen', 'testicle']
    vecs = importvec('./LiCO/' + '10_15_7path_addGB_4layerTrain_humanGRN_2_vec.txt')
    geneORGANizer = datas.importgeneItem()
    grn = datas.importgrnhg()
    ppi = datas.importppi()

    total_geneset = (set(geneORGANizer['genes'].keys()).union(set(ppi.keys()))).union(set(grn['genes'].keys()))
    print('total geneset length : ', len(total_geneset))

    prob = dict()
    bodypart_predict_bodypart_dict = dict()

    # geneORGANizer gene

    M = len(geneORGANizer['bodyparts'].keys())
    count = 0
    for organ in geneORGANizer['bodyparts'].keys():
        if organ in vecs:
            if organ not in bodypart_predict_bodypart_dict:
                bodypart_predict_bodypart_dict[organ] = set()
            prob[organ] = OrderedDict()
            # for gene in geneORGANizer['genes'].keys():
            # for gene in grn['genes'].keys():
            for gene in total_geneset:
                if gene in vecs:
                    prob[organ][gene] = cosine_similarity([vecs[gene], vecs[organ]])[0][1]

            vd = OrderedDict(sorted(prob[organ].items(), key=lambda t: t[1], reverse=True))
            i = 0
            for k in vd.keys():
                bodypart_predict_bodypart_dict[organ].add(k)
                i += 1
                if i == 2700:
                    break

            count = count + 1
            if count % 10 == 0:
                print(count, ' / ', M)
    return bodypart_predict_bodypart_dict


def generateDict_tf_predict_tg():
    vecs = importvec('./LiCO/' + '10_15_7path_addGB_4layerTrain_humanGRN_2_vec.txt')
    geneORGANizer = datas.importgeneItem()
    grn = datas.importgrnhg()
    ppi = datas.importppi()

    tf_predict_dict = dict()
    prob = dict()
    M = len(grn['tfs'].keys())
    count = 0

    for tf in grn['tfs'].keys():
        if tf not in tf_predict_dict:
            tf_predict_dict[tf] = set()
        prob[tf] = OrderedDict()
        for gene in grn['genes'].keys():
            if gene in vecs:
                prob[tf][gene] = cosine_similarity([vecs[tf], vecs[gene]])[0][1]

        vd = OrderedDict(sorted(prob[tf].items(), key=lambda t: t[1], reverse=True))
        i = 0
        for k in vd.keys():
            tf_predict_dict[tf].add(k)
            i += 1
            if i == 14000: # 因为包含了自身，故实际取11个
                break

        count = count + 1
        if count % 10 == 0:
            print(count, ' / ', M)
    return tf_predict_dict


def generateDict_tg_predict_tf():
    vecs = importvec('./LiCO/' + '10_15_7path_addGB_4layerTrain_humanGRN_2_vec.txt')
    geneORGANizer = datas.importgeneItem()
    grn = datas.import_database_grnhg()
    ppi = datas.importppi()

    tg_predict_dict = dict()
    prob = dict()
    M = len(grn['gene_tfs'].keys())
    count = 0

    count_tg = 0

    for tg in grn['gene_tfs'].keys():
        if tg in vecs:
            if tg not in tg_predict_dict:
                count_tg += 1
                tg_predict_dict[tg] = set()
            prob[tg] = OrderedDict()
            for tf in grn['tfs'].keys():
                if tf in vecs:
                    prob[tg][tf] = cosine_similarity([vecs[tf], vecs[tg]])[0][1]

            vd = OrderedDict(sorted(prob[tg].items(), key=lambda t: t[1], reverse=True))
            i = 0
            for k in vd.keys():
                tg_predict_dict[tg].add(k)
                i += 1
                if i == 418: # 因为包含了自身，故实际取11个
                    break

        count = count + 1
        if count % 10 == 0:
            print(count, ' / ', M)
    print('dict_predict_tg : ', count_tg)
    return tg_predict_dict


def lzm_calTopSim():
    M = len(diabetes_organs)
    prob = dict()
    count = 0
    # vecs = importvec('./model0.87/' + '6_12_0003_7path_17w_addGO140w_sz8(train_test5)_humanGRN_vec.txt')
    vecs = importvec('./LiCO/' + '10_15_7path_addGB_4layerTrain_humanGRN_2_vec.txt')

    geneORGANizer = datas.importgeneItem()
    grn = datas.importgrnhg()
    ppi = datas.importppi()


    # with open('./LiCO/human_predict/diabetes_organs_predict_top10genes.txt', 'w') as f:
    #     for organ in diabetes_organs:
    #         f.write(organ + '\t')
    #         prob[organ] = OrderedDict()
    #         for gene in geneORGANizer['genes'].keys():
    #             prob[organ][gene] = cosine_similarity([vecs[organ], vecs[gene]])[0][1]
    #
    #         vd = OrderedDict(sorted(prob[organ].items(), key=lambda t:t[1], reverse=True))
    #
    #         f.write(str(list(vd.items())[:10]))
    #         f.write('\n')
    #
    #         count = count + 1
    #         if count % 10 == 0:
    #             print(count, ' / ', M)
    #
    #
    #
    # with open('./LiCO/human_predict/diabetes_organs_predict_top10TFs.txt', 'w') as f:
    #     count = 0
    #     for organ in diabetes_organs:
    #         f.write(organ + '\t')
    #         prob[organ] = OrderedDict()
    #         for tf in grn['tfs'].keys():
    #             prob[organ][tf] = cosine_similarity([vecs[organ], vecs[tf]])[0][1]
    #
    #         vd = OrderedDict(sorted(prob[organ].items(), key=lambda t: t[1], reverse=True))
    #
    #         f.write(str(list(vd.items())[:10]))
    #         f.write('\n')
    #
    #         count = count + 1
    #         if count % 10 == 0:
    #             print(count, ' / ', M)
    #

    # diabetes_organs_predict_top10REs.txt
    # organs_predict_top10REs.txt
    # with open('./LiCO/human_predict/GTExorgans_predict_top10REs.txt', 'w') as f:
    #     count = 0
    #     for organ in GTEx_organs:
    #         f.write(organ + '\t')
    #         prob[organ] = OrderedDict()
    #         for re in grn['res'].keys():
    #             prob[organ][re] = cosine_similarity([vecs[organ], vecs[re]])[0][1]
    #
    #         vd = OrderedDict(sorted(prob[organ].items(), key=lambda t: t[1], reverse=True))
    #
    #         f.write(str(list(vd.items())[:10]))
    #         f.write('\n')
    #
    #         count = count + 1
    #         if count % 10 == 0:
    #             print(count, ' / ', M)

    # GRN gene
    # GRNgenes_predict_top10bodyparts.txt
    # with open('./LiCO/human_predict/GRNgenes_predict_top10organs.txt', 'w') as f:
    #     M = len(grn['genes'].keys())
    #     count = 0
    #     for gene in grn['genes'].keys():
    #         if gene in vecs:
    #             f.write(gene + '\t')
    #             prob[gene] = OrderedDict()
    #             for bodypart in gO_organs:
    #                 prob[gene][bodypart] = cosine_similarity([vecs[gene], vecs[bodypart]])[0][1]
    #
    #             vd = OrderedDict(sorted(prob[gene].items(), key=lambda t: t[1], reverse=True))
    #
    #             f.write(str(list(vd.items())[:10]))
    #             f.write('\n')
    #
    #             count = count + 1
    #             if count % 10 == 0:
    #                 print(count, ' / ', M)

    # PPI gene
    # PPIgenes_predict_top10bodyparts.txt
    # with open('./LiCO/human_predict/PPIgenes_predict_top10organs.txt', 'w') as f:
    #     M = len(ppi.keys())
    #     count = 0
    #     for gene in ppi.keys():
    #         if gene in vecs:
    #             f.write(gene + '\t')
    #             prob[gene] = OrderedDict()
    #             for bodypart in gO_organs:
    #                 prob[gene][bodypart] = cosine_similarity([vecs[gene], vecs[bodypart]])[0][1]
    #
    #             vd = OrderedDict(sorted(prob[gene].items(), key=lambda t: t[1], reverse=True))
    #
    #             f.write(str(list(vd.items())[:10]))
    #             f.write('\n')
    #
    #             count = count + 1
    #             if count % 10 == 0:
    #                 print(count, ' / ', M)


    # geneORGANizer gene
    # with open('./LiCO/human_predict/geneORGANizer_genes_predict_top10bpdyparts.txt', 'w') as f:
    #     M = len(ppi.keys())
    #     count = 0
    #     for gene in geneORGANizer['genes'].keys():
    #         if gene in vecs:
    #             f.write(gene + '\t')
    #             prob[gene] = OrderedDict()
    #             for bodypart in geneORGANizer['bodyparts'].keys():
    #                 prob[gene][bodypart] = cosine_similarity([vecs[gene], vecs[bodypart]])[0][1]
    #
    #             vd = OrderedDict(sorted(prob[gene].items(), key=lambda t: t[1], reverse=True))
    #
    #             f.write(str(list(vd.items())[:10]))
    #             f.write('\n')
    #
    #             count = count + 1
    #             if count % 10 == 0:
    #                 print(count, ' / ', M)
    #
    # # organ-GRN gene
    # with open('./LiCO/human_predict/organs_predict_top10GRN_genes.txt', 'w') as f:
    #     for organ in gO_organs:
    #         f.write(organ + '\t')
    #         prob[organ] = OrderedDict()
    #         for gene in grn['genes'].keys():
    #             if gene in vecs:
    #                 prob[organ][gene] = cosine_similarity([vecs[organ], vecs[gene]])[0][1]
    #
    #         vd = OrderedDict(sorted(prob[organ].items(), key=lambda t:t[1], reverse=True))
    #
    #         f.write(str(list(vd.items())[:10]))
    #         f.write('\n')
    #
    #         count = count + 1
    #         if count % 10 == 0:
    #             print(count, ' / ', M)
    #
    # # organ-PPI gene
    # with open('./LiCO/human_predict/organs_predict_top10PPI_genes.txt', 'w') as f:
    #     for organ in gO_organs:
    #         f.write(organ + '\t')
    #         prob[organ] = OrderedDict()
    #         for gene in ppi.keys():
    #             if gene in vecs:
    #                 prob[organ][gene] = cosine_similarity([vecs[organ], vecs[gene]])[0][1]
    #
    #         vd = OrderedDict(sorted(prob[organ].items(), key=lambda t: t[1], reverse=True))
    #
    #         f.write(str(list(vd.items())[:10]))
    #         f.write('\n')
    #
    #         count = count + 1
    #         if count % 10 == 0:
    #             print(count, ' / ', M)


    # # res_predict_top10bodyparts.txt
    # with open('./LiCO/human_predict/res_predict_top10organs.txt', 'w') as f:
    #     M = len(grn['res'].keys())
    #     count = 0
    #     for re in grn['res'].keys():
    #         f.write(re + '\t')
    #         prob[re] = OrderedDict()
    #         for bodypart in gO_organs:
    #             prob[re][bodypart] = cosine_similarity([vecs[re], vecs[bodypart]])[0][1]
    #
    #         vd = OrderedDict(sorted(prob[re].items(), key=lambda t:t[1], reverse=True))
    #
    #         f.write(str(list(vd.items())[:10]))
    #         f.write('\n')
    #
    #         count = count + 1
    #         if count % 10 == 0:
    #             print(count, ' / ', M)

    # tfs_predict_top10bodyparts.txt
    # tfs_predict_top10organs.txt
    with open('./LiCO/human_predict/tfs_predict_top10GRNgenes_2.txt', 'w') as f:
        tf_predict_dict = dict()

        M = len(grn['tfs'].keys())
        count = 0
        for tf in grn['tfs'].keys():
            if tf not in tf_predict_dict:
                tf_predict_dict[tf] = set()
            f.write(tf + '\t')
            prob[tf] = OrderedDict()
            for gene in grn['genes'].keys():
                if gene in vecs:
                    prob[tf][gene] = cosine_similarity([vecs[tf], vecs[gene]])[0][1]

            '''
            取出value值前n的key
            result = []
            result1 = []
            p = sorted([(k,v) for k, v in prob[tf].items()], reverse=True)
            s = set()
            for i in p:
                s.add(i[1])  # add value
            for i in sorted(s, reverse=True)[:n]: # 对取出的value进行排序，并取出前n个
                for j in p:
                    if j[1] == i:   # 如果原先的字典中值等于前n个value中的值， 则把这个key加入result
                        result.append(j)
            for r in result:
                result1.append(r[0]) # 取出前n个key
            '''

            vd = OrderedDict(sorted(prob[tf].items(), key=lambda t: t[1], reverse=True))

            # f.write(str(list(vd.items())[:10]))
            f.write(str(list(vd.keys())[:10]))
            f.write('\n')

            # tf_predict_dict[tf].add(vd.keys()[:10])
            i = 0
            for k in vd.keys():
                tf_predict_dict[tf].add(k)
                i += 1
                if i == 9:
                    break

            count = count + 1
            if count % 10 == 0:
                print(count, ' / ', M)
        return tf_predict_dict
    #
    # with open('./LiCO/human_predict/bodyparts_predict_top10bodyparts.txt', 'w') as f:
    #     M = len(geneORGANizer['bodyparts'].keys())
    #     count = 0
    #     for bodypartA in geneORGANizer['bodyparts'].keys():
    #         f.write(bodypartA + '\t')
    #         prob[bodypartA] = OrderedDict()
    #         for bodypartB in geneORGANizer['bodyparts'].keys():
    #             if bodypartB != bodypartA:
    #                 prob[bodypartA][bodypartB] = cosine_similarity([vecs[bodypartA], vecs[bodypartB]])[0][1]
    #
    #         vd = OrderedDict(sorted(prob[bodypartA].items(), key=lambda t:t[1], reverse=True))
    #
    #         f.write(str(list(vd.items())[:10]))
    #         f.write('\n')
    #
    #         count = count + 1
    #         if count % 10 == 0:
    #             print(count, ' / ', M)

    #
    # with open('./LiCO/human_predict/bodyparts_predict_top10res.txt', 'w') as f:
    #     M = len(geneORGANizer['bodyparts'].keys())
    #     count = 0
    #     for bodypart in geneORGANizer['bodyparts'].keys():
    #         f.write(bodypart + '\t')
    #         prob[bodypart] = OrderedDict()
    #         for re in grn['res'].keys():
    #             prob[bodypart][re] = cosine_similarity([vecs[bodypart], vecs[re]])[0][1]
    #
    #         vd = OrderedDict(sorted(prob[bodypart].items(), key=lambda t: t[1], reverse=True))
    #
    #         f.write(str(list(vd.items())[:10]))
    #         f.write('\n')
    #
    #         count = count + 1
    #         if count % 10 == 0:
    #             print(count, ' / ', M)

    # 聚类显示bodypart网络


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


def score_tf_predict_tg():
    # database_dict = datas.import_database_TRRUST()['tfs']

    tf_predict_dict = generateDict_tf_predict_tg()

    predict_fmax_tf_tg_dict = datas.import_predict_TF_TG_fmax()['tfs']

    # 用于检查grn中tf-tg关系和数据库的交集情况
    grn_dict = datas.import_database_grnhg()['tfs']

    database_hTFtarget_dict = datas.import_database_hTFtarget()['tfs']

    dict_diff(predict_fmax_tf_tg_dict, database_hTFtarget_dict, grn_dict)
    dict_2diff(predict_fmax_tf_tg_dict, database_hTFtarget_dict)


def score_tg_predict_tf():
    # database_dict = datas.import_database_TRRUST()['tgs']
    # print('databaseTRRUST tg : ', len(database_dict.keys()))

    tg_predict_dict = generateDict_tg_predict_tf()
    # print('tg_predict tg : ', len(tg_predict_dict.keys()))

    # 用于检查grn中tf-tg关系和数据库的交集情况
    grn_dict = datas.import_database_grnhg()['tfs']
    # print('grn_dict tg : ', len(grn_dict.keys()))

    database_hTFtarget_dict = datas.import_database_hTFtarget()['tfs']

    # count_overlap = 0
    # for key in database_hTFtarget_dict:
    #     if key in grn_dict:
    #         count_overlap += 1
    #
    # print(count_overlap)

    dict_diff(tg_predict_dict, database_hTFtarget_dict, grn_dict)
    dict_2diff(grn_dict, database_hTFtarget_dict)
    # dict_2diff(tg_predict_dict, database_dict)


def score_gene_predict_bodypart():
    # database_dict = datas.import_database_geneORGANizer()['genes']

    # databaseRNA_Altas_dict = datas.import_database_RNA_Atlas() # 该数据库仅包含11个组织

    database_hTFtarget_dict = datas.import_database_hTFtarget()['genes']

    # gene_predict_dict = generateDict_gene_predict_bodypart()

    database_cellTaxonomy_dict = datas.import_database_cellTaxonomy_excel()

    geneORGANizer = datas.importgeneItem()
    grn = datas.importgrnhg()
    ppi = datas.importppi()

    total_geneset = (set(geneORGANizer['genes'].keys()).union(set(ppi.keys()))).union(set(grn['genes'].keys()))



    count_overlap = 0
    for key1 in geneORGANizer['genes'].keys():
        key1 = key1.upper()
        for key2 in database_cellTaxonomy_dict['genes']:
            key2 = key2.upper()
            if key1 == key2:
                count_overlap += 1

    print(count_overlap)



    # my_dict_diff
    # dict_2diff(gene_predict_dict, database_dict)
    # dict_2diff(database_dict, database_hTFtarget_dict)
    # dict_diff(gene_predict_dict, databaseRNA_Altas_dict, database_dict)

def score_bodypart_predict_gene():
    database_dict = datas.import_database_geneORGANizer()['bodyparts']
    gene_database_dict = datas.import_database_geneORGANizer()['genes']


    databaseRNA_Altas_dict = datas.import_database_RNA_Atlas()['organs'] # 该数据库仅包含11个组织

    database_cellTaxonomy_dict = datas.import_database_cellTaxonomy_excel()

    normal_cellmaker = datas.import_database_cellMarker()['normal_organs']
    cancer_cellmaker = datas.import_database_cellMarker()['cancer_organs']


    # gene_predict_dict = generateDict_gene_predict_bodypart()

    # organ_predict_gene = generateDict_bodypart_predict_gene()

    # 根据阈值得到的bodypart-gene关系
    predict_fmax_b_g_dict = datas.import_predict_b_g_fmax()['bodyparts']

    # database_hTFtarget_dict = datas.import_database_hTFtarget()['tissue_genes']

    # 找重合
    cnt_overlap = 0
    for b in database_dict:
        if b in database_cellTaxonomy_dict['tissues'].keys():
            cnt_overlap += 1
    print('cnt_overlap: ', cnt_overlap)

    # my_dict_diff
    # dict_2diff(database_dict, database_cellTaxonomy_dict['tissues'])
    # dict_2diff(database_dict, databaseRNA_Altas_dict)
    # dict_diff(predict_fmax_b_g_dict, databaseRNA_Altas_dict, database_dict)

def diff_geneORGNizer_cellMarker():
    geneORGANizer_organ = datas.import_database_geneORGANizer_organ()['organs']


    normal_cellmaker = datas.import_database_cellMarker()['normal_organs']
    cancer_cellmaker = datas.import_database_cellMarker()['cancer_organs']

    # geneORGANizer_gene = datas.import_database_geneORGANizer()['genes']

    count_overlap = 0
    count_normal = 0
    count_cancer = 0

    for key in normal_cellmaker:
        count_normal += 1
        # count_cancer += 1
        if key in geneORGANizer_organ:
            count_overlap += 1
    print('overlap: ', count_overlap)
    print('count_normal: ', count_normal)
    # print('count_cancer: ', count_cancer)


    # dict_diff(geneORGANizer_organ, normal_cellmaker)

def dict_diff(first, second, third):
    """
    比较两个字典的不同
    third用于比对新预测出的部分的召回率、精确率、F1
    """
    # 召回率：找全率_
    # 精确率：找对率_检测成功基础上的正确率
    # F1：精确率和召回率的调和平均 = 2*pre*recall / (pre + recall)
    with open('./LiCO/human_predict/diff_newfmaxTFTG_hTFtarget.txt', 'w') as f:
        count = 0
        total_len1 = 0 # first
        total_len2 = 0 # second
        total_len_overlap = 0 # first、second交集
        total_len_new = 0 # 新预测出且不在third中

        for key in first:
            len_1 = 0
            len_2 = 0
            len_3 = 0
            f.write(key + '\t')
            set1 = first[key]
            # len_1 = len(set1) # dict1 key对应值的数量
            # total_len1 += len_1 # dict1 所有key对应的value数量
            if key in second:
                # 如果有重合，再统计len1
                len_1 = len(set1)
                total_len1 += len_1

                set2 = second[key]
                len_2 = len(set2) # dict2 key对应值的数量
                total_len2 += len_2 # dict2 所有key对应的value数量

                # overlap = set1 & set2
                # len_overlap = len(overlap) # 当前key交集的数量
                # total_len_overlap += len_overlap # 所有交集的数量


                set3 = third[key] # geneORGANizer
                len3 = len(set3)

                new = set1 - set3 # 预测结果且不在GRN中的
                len_new = len(new)
                total_len_new += len_new

                # 关注的是该差集 与 标准数据库 的关系

                overlap = new & set2
                len_overlap = len(overlap)
                total_len_overlap += len_overlap

                f.write(str(len_new) + ' in predict, ' + str(len_2) + ' in hTFtarget, ' + 'overlap : ' + str(len_overlap) + '\t')
                if len_overlap:
                    recall = len_overlap / len_2
                    precision = len_overlap / len_new
                    f1_score = (2 * recall * precision) / (precision + recall)
                    f.write('recall : ' + str(recall) + '\t')
                    f.write('precision : ' + str(precision) + '\t')
                    f.write('f1 score : ' + str(f1_score) + '\t')
                # f.write(str(new))
                f.write('\n')

            else:
                f.write('\n')
                count += 1

        f.write('total : ' + '\n')
        f.write('in predict have ' + str(total_len_new) + ', ' + 'hTFtarget have ' + str(total_len2) + ', ' + 'overlap : ' + str(total_len_overlap) + '\n')
        if total_len_overlap:
            recall = total_len_overlap / total_len2
            precision = total_len_overlap / total_len_new
            f1_score = (2 * recall * precision) / (precision + recall)
            f.write('recall : ' + str(recall) + '\t')
            f.write('precision : ' + str(precision) + '\t')
            f.write('f1 score : ' + str(f1_score) + '\t')
    print(str(count) + ', tf in predict and not in hTFtarget')

def dict_2diff(first, second):
    """
    比较两个字典的不同
    """
    # 召回率：找全率_
    # 精确率：找对率_检测成功基础上的正确率
    # F1：精确率和召回率的调和平均 = 2*pre*recall / (pre + recall)
    with open('./LiCO/about_cellTaxonomy/diff_geneORGANizer_cellTaxonomy.txt', 'w') as f:
        count = 0
        total_len1 = 0 # first
        total_len2 = 0 # second
        total_len_overlap = 0 # first、second交集  set1 & set2
        total_len_new = 0 # 新预测出且不在third中   set1 - set2

        vecs = importvec('./LiCO/' + '10_15_7path_addGB_4layerTrain_humanGRN_2_vec.txt')
        for key in first:
            if key in vecs:
                f.write(key + '\t')
                if key in second:
                    # 如果有重合，再统计len1
                    set1 = first[key]
                    len_1 = len(set1)
                    total_len1 += len_1

                    set2 = second[key]['gene']
                    len_2 = len(set2) # dict2 key对应值的数量
                    total_len2 += len_2 # dict2 所有key对应的value数量

                    overlap = set1 & set2
                    len_overlap = len(overlap) # 当前key交集的数量
                    total_len_overlap += len_overlap


                    # new = set1 - set2 # 预测结果且不在geneORGANizer中的
                    # len_new = len(new)
                    # total_len_new += len_new

                    # f.write(str(new) + '\t')

                    # f.write(str(len_1) + ' in dict1, ' + str(len_2) + ' in dict2, ' + 'new predict : ' + str(len_new))
                    # f.write(str(len_new) + ' in predictNew, ' + str(len_2) + ' in dict2, ' + '\t')
                    f.write(str(len_1) + ' in geneORGNizer, ' + str(len_2) + ' in cellTaxonomy, ' + 'overlap : ' + str(len_overlap))
                    f.write('\n')
                    if len_overlap:
                        recall = len_overlap / len_2
                        precision = len_overlap / len_1
                        f1_score = (2 * recall * precision) / (precision + recall)
                        f.write('recall : ' + str(recall) + '\t')
                        f.write('precision : ' + str(precision) + '\t')
                        f.write('f1 score : ' + str(f1_score) + '\t')
                        # f.write(str(new))
                        f.write('\n')

                else:
                    f.write('\n')
                    count += 1

        f.write('total : ' + '\n')
        # f.write(str(total_len1) + ' in dict1, ' + str(total_len2) + ' in dict2, ' + 'new predict : ' + str(total_len_new) + '\n')
        f.write(str(total_len1) + ' in geneORGNizer, ' + str(total_len2) + ' in cellTaxonomy, ' + 'overlap : ' + str(total_len_overlap) + '\n')
        if total_len_overlap:
            recall = total_len_overlap / total_len2
            precision = total_len_overlap / total_len1
            f1_score = (2 * recall * precision) / (precision + recall)
            f.write('recall : ' + str(recall) + '\t')
            f.write('precision : ' + str(precision) + '\t')
            f.write('f1 score : ' + str(f1_score) + '\t')
        f.write(str(count) + ' tissue in geneORGNizer and not in cellTaxonomy')

def main():
    # 糖尿病相关organ
    global geneORGANizer
    # geneORGANizer = datas.importgene_organs('./New_gene_organs_2.txt')
    geneORGANizer = datas.importgeneItem()

    vecs = importvec('./model0.87/' + '6_12_0003_7path_17w_addGO140w_sz8(train_test5)_humanGRN_vec.txt')

    bodyparts = geneORGANizer['bodyparts'].keys()

    # res = res[6000:7000]
    # fmax, organs = getthres()
    # calres(vecs, res, high, fmax, low, terms)

    redict = Manager().dict()
    N = 8
    bodypartsset = []
    for i in range(N):
        bodypartsset.append(set())
    for i in bodyparts:
        bodypartsset[random.randint(0, N - 1)].add(i)
    processes = []
    for i in range(N):
        processes.append(Process(target=lzm_calbodyparts, args=(redict, vecs, bodypartsset[i], geneORGANizer, 'p' + str(i))))
    for i in range(N):
        processes[i].start()
    for i in range(N):
        processes[i].join()
    with open('./model0.87/predict/' + 'bodypart-bodypart-0.8.txt', 'w') as f:  # 'a' 文件不存在也会创建写入
        for re in redict.keys():
            f.write(re + '\t' + '\t'.join(list(redict[re]))+'\n')

def test():
    array = [10, 17, 50, 7, 30, 24, 27, 45, 15, 5, 36, 21]
    heapq.heapify(array)
    print(heapq.nlargest(2, array))
    print(heapq.nsmallest(3, array))


def calscore():
    vecs = importvec('./LiCO/' + '10_15_7path_addGB_4layerTrain_humanGRN_2_vec.txt')
    geneORGANizer = datas.importgeneItem()
    grn = datas.import_database_grnhg()
    ppi = datas.importppi()

    total_geneset = (set(geneORGANizer['genes'].keys()).union(set(ppi.keys()))).union(set(grn['gene_tfs'].keys()))



    # 验证数据库
    tf_database_hTFtarget_dict = datas.import_database_hTFtarget()['tfs']
    tf_database_TRRUST_dict = datas.import_database_TRRUST()['tfs']

    gene_database_hTFtarget_dict = datas.import_database_hTFtarget()['genes']
    gene_database_TRRUST_dict = datas.import_database_TRRUST()['tgs']

    tf_set1 = set(grn['tfs']).union(tf_database_hTFtarget_dict.keys())
    tf_set2 = set(grn['tfs']).union(tf_database_TRRUST_dict.keys())


    # 以重合的TF->TG
    # 给每个TF确定一个阈值：每个TF与所有TG计算相似度，如果在数据库中存在关系，记为1，反之为0；最后得到PR曲线
    standard = []
    score = []

    count_one = 0
    count_zero = 0
    for tf in tf_set1:
        print('len of tfs : ', len(tf_set1))
        if tf in vecs and tf in tf_database_hTFtarget_dict.keys(): # 选取重合的tf
            for gene in total_geneset:
                if gene in vecs:
                    ans = {cosine_similarity([vecs[tf], vecs[gene]])[0][1]}
                    ans = list(ans)[0]
                    score.append(ans)
                    # 如果关系被验证数据库验证，则1；反之0
                    if gene in tf_database_hTFtarget_dict[tf]:
                        count_one += 1
                        standard.append(1)
                    else:
                        count_zero += 1
                        standard.append(0)

    precision, recall, thresholds = precision_recall_curve(standard, score)

    with open('./LiCO/fig/p-r_TF-TG_hTFtarget.txt', 'w') as f:
        f.write('one : ' + str(count_one) + '\n')
        f.write('zero : ' + str(count_zero) + '\n')
        f.write('precision : ' + str(precision) + '\n')
        f.write('recall : ' + str(recall) + '\n')
        f.write('thresholds : ' + str(thresholds) + '\n')


    plt.figure(1)  # 创建图表1
    plt.title('Precision/Recall Curve')  # give plot a title
    plt.xlabel('Recall')  # make axis labels
    plt.ylabel('Precision')

    plt.figure(1)
    plt.plot(precision, recall)
    plt.show()
    plt.savefig("./LiCO/fig/p-r_TF-TG_hTFtarget" + ".jpg")

    # 以重合的TG->TF
    count_one = 0
    count_zero = 0
    for gene in total_geneset:
        if gene in vecs and gene in gene_database_hTFtarget_dict.keys(): # 选取重合的G
            for tf in tf_set1:
                if tf in vecs:
                    ans = {cosine_similarity([vecs[tf], vecs[gene]])[0][1]}
                    ans = list(ans)[0]
                    score.append(ans)
                    # 如果关系被验证数据库验证，则1；反之0
                    if tf in gene_database_hTFtarget_dict[gene]:
                        count_one += 1
                        standard.append(1)
                    else:
                        count_zero += 1
                        standard.append(0)


    precision, recall, thresholds = precision_recall_curve(standard, score)

    with open('./LiCO/fig/p-r_TG-TF_hTFtarget.txt', 'w') as f:
        f.write('one : ' + str(count_one) + '\n')
        f.write('zero : ' + str(count_zero) + '\n')
        f.write('precision : ' + str(precision) + '\n')
        f.write('recall : ' + str(recall) + '\n')
        f.write('thresholds : ' + str(thresholds) + '\n')


    plt.figure(1)  # 创建图表1
    plt.title('Precision/Recall Curve')  # give plot a title
    plt.xlabel('Recall')  # make axis labels
    plt.ylabel('Precision')

    plt.figure(1)
    plt.plot(precision, recall)
    plt.show()
    plt.savefig('./LiCO/fig/p-r_TG-TF_hTFtarget.jpg')


def cal_box_tftg():
    vecs = importvec('./LiCO/' + '10_15_7path_addGB_4layerTrain_humanGRN_2_vec.txt')
    geneORGANizer = datas.importgeneItem()
    grn = datas.import_database_grnhg()
    ppi = datas.importppi()

    total_geneset = (set(geneORGANizer['genes'].keys()).union(set(ppi.keys()))).union(set(grn['gene_tfs'].keys()))

    tf_database_hTFtarget_dict = datas.import_database_hTFtarget()['tfs']
    tf_database_TRRUST_dict = datas.import_database_TRRUST()['tfs']

    gene_database_hTFtarget_dict = datas.import_database_hTFtarget()['genes']
    gene_database_TRRUST_dict = datas.import_database_TRRUST()['tgs']

    database_TRRUST_dict = datas.import_database_TRRUST()
    database_hTFtarget_dict = datas.import_database_hTFtarget()

    # 根据fmax阈值得到的TF-TG关系
    predict_fmax_tf_tg_dict = datas.import_predict_TF_TG_fmax()

    # 取grn和标准数据库的交集 不能取并集 因为是TF预测TG
    tf_set1 = set(grn['tfs']).intersection(tf_database_hTFtarget_dict.keys())
    tf_set2 = set(grn['tfs']).intersection(tf_database_TRRUST_dict.keys())

    # Boxplot test

    in_dict2 = []
    not_in_dict2 = []

    count_one = 0
    count_zero = 0
    # for tf in tf_set1:
    #     if tf in vecs and tf in tf_database_hTFtarget_dict.keys(): # 选取重合的tf
    #         for gene in total_geneset:
    #             if gene in vecs:
    #                 ans = {cosine_similarity([vecs[tf], vecs[gene]])[0][1]}
    #                 ans = list(ans)[0]
    #                 # 如果关系被验证数据库验证，则值加入in_dict2
    #                 if gene in tf_database_hTFtarget_dict[tf]:
    #                     count_one += 1
    #                     in_dict2.append(ans)
    #                 else:
    #                     count_zero += 1
    #                     not_in_dict2.append(ans)

    # GRN和2个数据的TF-TG关系的分布
    grn_tftg = []
    for tf in grn['tfs']:
        if tf in vecs:
            for gene in grn['tfs'][tf]:
                if gene in vecs:
                    ans = {cosine_similarity([vecs[tf], vecs[gene]])[0][1]}
                    ans = list(ans)[0]
                    grn_tftg.append(ans)

    TRRUST_tftg = []
    for tf in database_TRRUST_dict['tfs']:
        if tf in vecs:
            for gene in database_TRRUST_dict['tfs'][tf]:
                if gene in vecs:
                    ans = {cosine_similarity([vecs[tf], vecs[gene]])[0][1]}
                    ans = list(ans)[0]
                    TRRUST_tftg.append(ans)

    hTFtarget_tftg = []
    for tf in database_hTFtarget_dict['tfs']:
        if tf in vecs:
            for gene in database_hTFtarget_dict['tfs'][tf]:
                if gene in vecs:
                    ans = {cosine_similarity([vecs[tf], vecs[gene]])[0][1]}
                    ans = list(ans)[0]
                    hTFtarget_tftg.append(ans)



    # 新预测的 不在GRN也不在参考数据库的 vs 不在GRN但是在参考数据库中的
    count_1 = 0
    count_2 = 0
    not_in_grn_hTFtarget = []
    not_in_grn_but_in_hTFtarget = []
    # not_in_grn_TRRUST = []
    # not_in_grn_but_in_TRRUST = []

    for tf in predict_fmax_tf_tg_dict['tfs']:
        # 如果该tf在GRN和hTFtarget的TF并集中(set1)、TRRUST(set2)
        if tf in tf_set1:
            # 遍历预测出的tg，挑出不在GRN且不在参考数据集 和 不在GRN但在参考数据集中的
            for tg in predict_fmax_tf_tg_dict['tfs'][tf]:
                if tg in vecs:
                    if tg not in grn['tfs'][tf] and tg not in database_hTFtarget_dict['tfs'][tf]:
                        count_1 += 1
                        ans = {cosine_similarity([vecs[tf], vecs[tg]])[0][1]}
                        ans = list(ans)[0]
                        not_in_grn_hTFtarget.append(ans)
                    if tg not in grn['tfs'][tf] and tg in database_hTFtarget_dict['tfs'][tf]:
                        count_2 += 1
                        ans = {cosine_similarity([vecs[tf], vecs[tg]])[0][1]}
                        ans = list(ans)[0]
                        not_in_grn_but_in_hTFtarget.append(ans)

    print('in database : ', count_2)
    print('not in database : ', count_1)

    # 做标准化
    # z-score
    zscore_scaler = preprocessing.StandardScaler()
    not_in_grn_hTFtarget = np.array(not_in_grn_hTFtarget).reshape(-1, 1)  # 1规定列数为1，-1表根据给定的列数自动分配行数
    not_in_grn_but_in_hTFtarget = np.array(not_in_grn_but_in_hTFtarget).reshape(-1, 1)


    temp1 = np.array(not_in_grn_hTFtarget).reshape(1, -1)
    temp1 = temp1.tolist()
    levene_1 = temp1[0]

    temp2 = np.array(not_in_grn_but_in_hTFtarget).reshape(1, -1)
    temp2 = temp2.tolist()
    levene_2 = temp2[0]

    # 方差齐性检验
    stat, p = levene(levene_1, levene_2)
    print('raw_levene:')
    print(stat, p)

    # 方差无显著差异
    if p >= 0.05:
        t, pval = t_test(not_in_grn_hTFtarget, not_in_grn_but_in_hTFtarget)
        print('raw:')
        print(t, pval)
    # 方差有显著差异
    else:
        print("equal_vasr=False")
        t, pval = t_test(not_in_grn_hTFtarget, not_in_grn_but_in_hTFtarget, False)
        print('raw:')
        print(t, pval)


    data_score_1 = zscore_scaler.fit_transform(not_in_grn_hTFtarget)
    data_score_2 = zscore_scaler.fit_transform(not_in_grn_but_in_hTFtarget)

    temp1 = np.array(data_score_1).reshape(1, -1)
    temp1 = temp1.tolist()
    levene_1 = temp1[0]

    temp2 = np.array(data_score_2).reshape(1, -1)
    temp2 = temp2.tolist()
    levene_2 = temp2[0]


    # 方差齐性检验
    stat, p = levene(levene_1, levene_2)
    print('zscore_levene:')
    print(stat, p)

    # 方差无显著差异
    if p >= 0.05:
        t, pval = t_test(data_score_1, data_score_2)
        print('zscore:')
        print(t, pval)
    # 方差有显著差异
    else:
        print("equal_vasr=False")
        t, pval = t_test(data_score_1, data_score_2, False)
        print('zscore:')
        print(t, pval)


    # print('original: ', not_in_grn_TFtarget)
    # print('transformed: ', data_score_1)

    # # max-min
    minmax_scaler = preprocessing.MinMaxScaler()
    data_minmax_1 = minmax_scaler.fit_transform(not_in_grn_hTFtarget)
    data_minmax_2 = minmax_scaler.fit_transform(not_in_grn_but_in_hTFtarget)

    temp1 = np.array(data_minmax_1).reshape(1, -1)
    temp1 = temp1.tolist()
    levene_1 = temp1[0]

    temp2 = np.array(data_minmax_2).reshape(1, -1)
    temp2 = temp2.tolist()
    levene_2 = temp2[0]

    # 方差齐性检验
    stat, p = levene(levene_1, levene_2)
    print('minmax_levene:')
    print(stat, p)

    # 方差无显著差异
    if p >= 0.05:
        t, pval = t_test(data_minmax_1, data_minmax_2)
        print('minmax:')
        print(t, pval)
    # 方差有显著差异
    else:
        print("equal_vasr=False")
        t, pval = t_test(data_minmax_1, data_minmax_2, False)
        print('minmax:')
        print(t, pval)

    # maxAbs
    maxabs_scaler = preprocessing.MaxAbsScaler()
    data_maxabs_1 = maxabs_scaler.fit_transform(not_in_grn_hTFtarget)
    data_maxabs_2 = maxabs_scaler.fit_transform(not_in_grn_but_in_hTFtarget)

    temp1 = np.array(data_maxabs_1).reshape(1, -1)
    temp1 = temp1.tolist()
    levene_1 = temp1[0]

    temp2 = np.array(data_maxabs_2).reshape(1, -1)
    temp2 = temp2.tolist()
    levene_2 = temp2[0]

    # 方差齐性检验
    stat, p = levene(levene_1, levene_2)
    print('maxabs_levene:')
    print(stat, p)

    # 方差无显著差异
    if p >= 0.05:
        t, pval = t_test(data_maxabs_1, data_maxabs_2)
        print('maxabs:')
        print(t, pval)
    # 方差有显著差异
    else:
        print("equal_vasr=False")
        t, pval = t_test(data_maxabs_1, data_maxabs_2, False)
        print('maxabs:')
        print(t, pval)


    # robustScaler
    Robust_scaler = preprocessing.RobustScaler()
    data_Robust_1 = Robust_scaler.fit_transform(not_in_grn_hTFtarget)
    data_Robust_2 = Robust_scaler.fit_transform(not_in_grn_but_in_hTFtarget)

    temp1 = np.array(data_Robust_1).reshape(1, -1)
    temp1 = temp1.tolist()
    levene_1 = temp1[0]

    temp2 = np.array(data_Robust_2).reshape(1, -1)
    temp2 = temp2.tolist()
    levene_2 = temp2[0]

    # 方差齐性检验
    stat, p = levene(levene_1, levene_2)
    print('Robust_levene:')
    print(stat, p)

    # 方差无显著差异
    if p >= 0.05:
        t, pval = t_test(data_Robust_1, data_Robust_2)
        print('Robust:')
        print(t, pval)
    # 方差有显著差异
    else:
        print("equal_vasr=False")
        t, pval = t_test(data_Robust_1, data_Robust_2, False)
        print('Robust:')
        print(t, pval)


    # # Normalizer
    # Normal_l1_scaler = preprocessing.Normalizer(norm='l1')
    # data_Normal_l1_1 = Normal_l1_scaler.fit_transform(not_in_grn_TFtarget)
    # data_Normal_l1_2 = Normal_l1_scaler.fit_transform(not_in_grn_but_in_TFtarget)
    #
    # Normal_l2_scaler = preprocessing.Normalizer(norm='l2')
    # data_Normal_l2_1 = Normal_l2_scaler.fit_transform(not_in_grn_TFtarget)
    # data_Normal_l2_2 = Normal_l2_scaler.fit_transform(not_in_grn_but_in_TFtarget)
    #
    # Normal_max_scaler = preprocessing.Normalizer(norm='max')
    # data_Normal_max_1 = Normal_max_scaler.fit_transform(not_in_grn_TFtarget)
    # data_Normal_max_2 = Normal_max_scaler.fit_transform(not_in_grn_but_in_TFtarget)





    # print('count_one : ', count_one)
    # print('count_zero : ', count_zero)
    # 绘制箱线图
    labels = 'raw_1', 'raw_2', \
             'z_1', 'z_2', \
             'minmax_1', 'minmax_2', \
             'maxabs_1', 'maxabs_2', \
             'Robust_1', 'Robust_2'
    plt.grid(True) # 显示网格
    plt.boxplot([not_in_grn_hTFtarget, not_in_grn_but_in_hTFtarget,
                 data_score_1, data_score_2,
                 data_minmax_1, data_minmax_2,
                 data_maxabs_1, data_maxabs_2,
                 data_Robust_1, data_Robust_2
                 ],
                medianprops={'color': 'red', 'linewidth': '1.5'},
                meanline=True,
                showmeans=True,
                meanprops={'color': 'blue', 'ls': '--', 'linewidth': '1.5'},
                flierprops={"marker": "o", "markerfacecolor": "red", "markersize": 5},
                labels=labels)
    plt.yticks(np.arange(-2, 5.5, 0.5))
    plt.savefig('./LiCO/fig/box_tftg_hTFtarget_not_in_vs_in.png', dpi=350)
    plt.show()


def cal_box_g_b():
    vecs = importvec('./LiCO/' + '10_15_7path_addGB_4layerTrain_humanGRN_2_vec.txt')
    geneORGANizer = datas.importgeneItem()
    grn = datas.import_database_grnhg()
    ppi = datas.importppi()

    total_geneset = (set(geneORGANizer['genes'].keys()).union(set(ppi.keys()))).union(set(grn['gene_tfs'].keys()))


    # 参考数据库RNA-Altas
    bodypart_database_RNAAltas_dict = datas.import_database_RNA_Atlas()['organs']
    gene_database_RNAAltas_dict = datas.import_database_RNA_Atlas()['genes']

    # 参考数据库hTFtarget
    bodypart_database_hTFtarget_dict = datas.import_database_hTFtarget()['tissue_genes']

    # 参考数据库GTEx


    # 根据fmax阈值得到的b-g关系
    predict_fmax_b_g_dict = datas.import_predict_b_g_fmax()


    # Boxplot test
    # 一、三个数据库自身的分布
    # geneORGANizer和3个数据库的B-G关系的分布
    # gO_B_G = []
    # for bodypart in geneORGANizer['bodyparts']:
    #     if bodypart in vecs:
    #         for gene in geneORGANizer['bodyparts'][bodypart]:
    #             if gene in vecs:
    #                 ans = {cosine_similarity([vecs[bodypart], vecs[gene]])[0][1]}
    #                 ans = list(ans)[0]
    #                 gO_B_G.append(ans)
    #
    # RNAAltas_B_G = []
    # for bodypart in bodypart_database_RNAAltas_dict:
    #     if bodypart in vecs:
    #         for gene in bodypart_database_RNAAltas_dict[bodypart]:
    #             if gene in vecs:
    #                 ans = {cosine_similarity([vecs[bodypart], vecs[gene]])[0][1]}
    #                 ans = list(ans)[0]
    #                 RNAAltas_B_G.append(ans)
    #
    # hTFtarget_B_G = []
    # for bodypart in bodypart_database_hTFtarget_dict:
    #     if bodypart in vecs:
    #         for gene in bodypart_database_hTFtarget_dict[bodypart]:
    #             if gene in vecs:
    #                 ans = {cosine_similarity([vecs[bodypart], vecs[gene]])[0][1]}
    #                 ans = list(ans)[0]
    #                 hTFtarget_B_G.append(ans)

    # 二、新预测的数据集
    # 新预测的 不在GRN也不在参考数据库的 vs 不在GRN但是在参考数据库中的
    count_not_in_gO_RNAAltas = 0
    count_not_in_gO_but_in_RNAAltas = 0

    not_in_gO_RNAAltas = []
    not_in_gO_but_in_RNAAltas = []

    for bodypart in predict_fmax_b_g_dict['bodyparts']:
        # bodypart必在vecs中，这里要保证在参考数据库中
        if bodypart in bodypart_database_RNAAltas_dict:
            # 遍历预测出的tg，挑出不在GRN且不在参考数据集 和 不在GRN但在参考数据集中的
            for gene in predict_fmax_b_g_dict['bodyparts'][bodypart]:
                if gene in vecs:
                    if gene not in geneORGANizer['bodyparts'][bodypart] and gene not in bodypart_database_RNAAltas_dict[bodypart]:
                        count_not_in_gO_RNAAltas += 1
                        ans = {cosine_similarity([vecs[bodypart], vecs[gene]])[0][1]}
                        ans = list(ans)[0]
                        not_in_gO_RNAAltas.append(ans)
                    if gene not in geneORGANizer['bodyparts'][bodypart] and gene in bodypart_database_RNAAltas_dict[bodypart]:
                        count_not_in_gO_but_in_RNAAltas += 1
                        ans = {cosine_similarity([vecs[bodypart], vecs[gene]])[0][1]}
                        ans = list(ans)[0]
                        not_in_gO_but_in_RNAAltas.append(ans)

    print('in database : ', count_not_in_gO_but_in_RNAAltas)
    print('not in database : ', count_not_in_gO_RNAAltas)



    # count_not_in_gO_hTFtarget = 0
    # count_not_in_gO_but_in_hTFtarget = 0
    #
    # not_in_gO_hTFtarget = []
    # not_in_gO_but_in_hTFtarget = []
    #
    # for bodypart in predict_fmax_b_g_dict['bodyparts']:
    #     if bodypart in bodypart_database_hTFtarget_dict:
    #         # 遍历预测出的tg，挑出不在GRN且不在参考数据集 和 不在GRN但在参考数据集中的
    #         for gene in predict_fmax_b_g_dict['bodyparts'][bodypart]:
    #             if gene in vecs:
    #                 if gene not in geneORGANizer['bodyparts'][bodypart] and gene not in bodypart_database_hTFtarget_dict[bodypart]:
    #                     count_not_in_gO_hTFtarget += 1
    #                     ans = {cosine_similarity([vecs[bodypart], vecs[gene]])[0][1]}
    #                     ans = list(ans)[0]
    #                     not_in_gO_hTFtarget.append(ans)
    #                 if gene not in geneORGANizer['bodyparts'][bodypart] and gene in bodypart_database_hTFtarget_dict[bodypart]:
    #                     count_not_in_gO_but_in_hTFtarget += 1
    #                     ans = {cosine_similarity([vecs[bodypart], vecs[gene]])[0][1]}
    #                     ans = list(ans)[0]
    #                     not_in_gO_but_in_hTFtarget.append(ans)
    #
    # print('in database : ', count_not_in_gO_but_in_hTFtarget)
    # print('not in database : ', count_not_in_gO_hTFtarget)

    # 做标准化
    # z-score
    zscore_scaler = preprocessing.StandardScaler()
    not_in_gO_RNAAltas = np.array(not_in_gO_RNAAltas).reshape(-1, 1)  # 1规定列数为1，-1表根据给定的列数自动分配行数
    not_in_gO_but_in_RNAAltas = np.array(not_in_gO_but_in_RNAAltas).reshape(-1, 1)

    temp1 = np.array(not_in_gO_RNAAltas).reshape(1, -1)
    temp1 = temp1.tolist()
    levene_1 = temp1[0]

    temp2 = np.array(not_in_gO_but_in_RNAAltas).reshape(1, -1)
    temp2 = temp2.tolist()
    levene_2 = temp2[0]

    # 方差齐性检验
    stat, p = levene(levene_1, levene_2)
    print('raw_levene:')
    print(stat, p)

    # 方差无显著差异
    if p >= 0.05:
        t, pval = t_test(not_in_gO_RNAAltas, not_in_gO_but_in_RNAAltas)
        print('raw:')
        print(t, pval)
    # 方差有显著差异
    else:
        print("equal_vasr=False")
        t, pval = t_test(not_in_gO_RNAAltas, not_in_gO_but_in_RNAAltas, False)
        print('raw:')
        print(t, pval)


    data_score_1 = zscore_scaler.fit_transform(not_in_gO_RNAAltas)
    data_score_2 = zscore_scaler.fit_transform(not_in_gO_but_in_RNAAltas)

    temp1 = np.array(data_score_1).reshape(1, -1)
    temp1 = temp1.tolist()
    levene_1 = temp1[0]

    temp2 = np.array(data_score_2).reshape(1, -1)
    temp2 = temp2.tolist()
    levene_2 = temp2[0]

    # 方差齐性检验
    stat, p = levene(levene_1, levene_2)
    print('zscore_levene:')
    print(stat, p)

    # 方差无显著差异
    if p >= 0.05:
        t, pval = t_test(data_score_1, data_score_2)
        print('zscore:')
        print(t, pval)
    # 方差有显著差异
    else:
        print("equal_vasr=False")
        t, pval = t_test(data_score_1, data_score_2, False)
        print('zscore:')
        print(t, pval)

    # print('original: ', not_in_grn_TFtarget)
    # print('transformed: ', data_score_1)

    # # max-min
    minmax_scaler = preprocessing.MinMaxScaler()
    data_minmax_1 = minmax_scaler.fit_transform(not_in_gO_RNAAltas)
    data_minmax_2 = minmax_scaler.fit_transform(not_in_gO_but_in_RNAAltas)

    temp1 = np.array(data_minmax_1).reshape(1, -1)
    temp1 = temp1.tolist()
    levene_1 = temp1[0]

    temp2 = np.array(data_minmax_2).reshape(1, -1)
    temp2 = temp2.tolist()
    levene_2 = temp2[0]

    # 方差齐性检验
    stat, p = levene(levene_1, levene_2)
    print('minmax_levene:')
    print(stat, p)

    # 方差无显著差异
    if p >= 0.05:
        t, pval = t_test(data_minmax_1, data_minmax_2)
        print('minmax:')
        print(t, pval)
    # 方差有显著差异
    else:
        print("equal_vasr=False")
        t, pval = t_test(data_minmax_1, data_minmax_2, False)
        print('minmax:')
        print(t, pval)

    # maxAbs
    maxabs_scaler = preprocessing.MaxAbsScaler()
    data_maxabs_1 = maxabs_scaler.fit_transform(not_in_gO_RNAAltas)
    data_maxabs_2 = maxabs_scaler.fit_transform(not_in_gO_but_in_RNAAltas)

    temp1 = np.array(data_maxabs_1).reshape(1, -1)
    temp1 = temp1.tolist()
    levene_1 = temp1[0]

    temp2 = np.array(data_maxabs_2).reshape(1, -1)
    temp2 = temp2.tolist()
    levene_2 = temp2[0]

    # 方差齐性检验
    stat, p = levene(levene_1, levene_2)
    print('maxabs_levene:')
    print(stat, p)

    # 方差无显著差异
    if p >= 0.05:
        t, pval = t_test(data_maxabs_1, data_maxabs_2)
        print('maxabs:')
        print(t, pval)
    # 方差有显著差异
    else:
        print("equal_vasr=False")
        t, pval = t_test(data_maxabs_1, data_maxabs_2, False)
        print('maxabs:')
        print(t, pval)

    # robustScaler
    Robust_scaler = preprocessing.RobustScaler()
    data_Robust_1 = Robust_scaler.fit_transform(not_in_gO_RNAAltas)
    data_Robust_2 = Robust_scaler.fit_transform(not_in_gO_but_in_RNAAltas)

    temp1 = np.array(data_Robust_1).reshape(1, -1)
    temp1 = temp1.tolist()
    levene_1 = temp1[0]

    temp2 = np.array(data_Robust_2).reshape(1, -1)
    temp2 = temp2.tolist()
    levene_2 = temp2[0]

    # 方差齐性检验
    stat, p = levene(levene_1, levene_2)
    print('Robust_levene:')
    print(stat, p)

    # 方差无显著差异
    if p >= 0.05:
        t, pval = t_test(data_Robust_1, data_Robust_2)
        print('Robust:')
        print(t, pval)
    # 方差有显著差异
    else:
        print("equal_vasr=False")
        t, pval = t_test(data_Robust_1, data_Robust_2, False)
        print('Robust:')
        print(t, pval)

    # # Normalizer
    # Normal_l1_scaler = preprocessing.Normalizer(norm='l1')
    # data_Normal_l1_1 = Normal_l1_scaler.fit_transform(not_in_gO_RNAAltas)
    # data_Normal_l1_2 = Normal_l1_scaler.fit_transform(not_in_gO_but_in_RNAAltas)
    #
    # Normal_l2_scaler = preprocessing.Normalizer(norm='l2')
    # data_Normal_l2_1 = Normal_l2_scaler.fit_transform(not_in_gO_RNAAltas)
    # data_Normal_l2_2 = Normal_l2_scaler.fit_transform(not_in_gO_but_in_RNAAltas)
    #
    # Normal_max_scaler = preprocessing.Normalizer(norm='max')
    # data_Normal_max_1 = Normal_max_scaler.fit_transform(not_in_gO_RNAAltas)
    # data_Normal_max_2 = Normal_max_scaler.fit_transform(not_in_gO_but_in_RNAAltas)

    # print('count_one : ', count_one)
    # print('count_zero : ', count_zero)
    # 绘制箱线图
    labels = 'raw_1', 'raw_2', \
             'z_1', 'z_2', \
             'minmax_1', 'minmax_2', \
             'maxabs_1', 'maxabs_2', \
             'Robust_1', 'Robust_2'

    plt.grid(True)  # 显示网格
    plt.boxplot([not_in_gO_RNAAltas, not_in_gO_but_in_RNAAltas,
                 data_score_1, data_score_2,
                 data_minmax_1, data_minmax_2,
                 data_maxabs_1, data_maxabs_2,
                 data_Robust_1, data_Robust_2],
                medianprops={'color': 'red', 'linewidth': '1.5'},
                meanline=True,
                showmeans=True,
                meanprops={'color': 'blue', 'ls': '--', 'linewidth': '1.5'},
                flierprops={"marker": "o", "markerfacecolor": "red", "markersize": 5},
                labels=labels)
    plt.yticks(np.arange(-4.4, 3.4, 0.2))
    plt.savefig('./LiCO/fig/box_gb_RNAAltas_not_in_vs_in_.jpg', dpi=350)
    plt.show()

# 做每一个B的sim标准胡
def cal_box_eachB_G():
    vecs = importvec('./LiCO/' + '10_15_7path_addGB_4layerTrain_humanGRN_2_vec.txt')
    geneORGANizer = datas.importgeneItem()
    grn = datas.import_database_grnhg()
    ppi = datas.importppi()

    total_geneset = (set(geneORGANizer['genes'].keys()).union(set(ppi.keys()))).union(set(grn['gene_tfs'].keys()))

    # 参考数据库RNA-Altas
    bodypart_database_RNAAltas_dict = datas.import_database_RNA_Atlas()['organs']
    gene_database_RNAAltas_dict = datas.import_database_RNA_Atlas()['genes']

    # 参考数据库hTFtarget
    bodypart_database_hTFtarget_dict = datas.import_database_hTFtarget()['tissue_genes']

    # 参考数据库GTEx

    # 根据fmax阈值得到的b-g关系
    predict_fmax_b_g_dict = datas.import_predict_b_g_fmax()

    cnt = 0
    raw_totalB_not_in = []
    raw_totalB_in = []
    zscore_totalB_not_in = []
    zscore_totalB_in = []
    minmax_totalB_not_in = []
    minmax_totalB_in = []
    maxabs_totalB_not_in = []
    maxabs_totalB_in = []
    Robust_totalB_not_in = []
    Robust_totalB_in = []
    # 为重合的每一个B做标准化
    for bodypart in predict_fmax_b_g_dict['bodyparts']:
        # bodypart必在vecs中，这里要保证在参考数据库中
        if bodypart in bodypart_database_RNAAltas_dict:
            count_not_in_gO_RNAAltas = 0
            count_not_in_gO_but_in_RNAAltas = 0
            not_in_gO_RNAAltas = []
            not_in_gO_but_in_RNAAltas = []
            # 针对每一个B，做内部的比对：不在参考数据库 VS 在参考数据库
            # 遍历预测出的tg，挑出不在GRN且不在参考数据集 和 不在GRN但在参考数据集中的
            for gene in predict_fmax_b_g_dict['bodyparts'][bodypart]:
                if gene in vecs:
                    if gene not in geneORGANizer['bodyparts'][bodypart] and gene not in bodypart_database_RNAAltas_dict[bodypart]:
                        count_not_in_gO_RNAAltas += 1
                        ans = {cosine_similarity([vecs[bodypart], vecs[gene]])[0][1]}
                        ans = list(ans)[0]
                        not_in_gO_RNAAltas.append(ans)
                    if gene not in geneORGANizer['bodyparts'][bodypart] and gene in bodypart_database_RNAAltas_dict[bodypart]:
                        count_not_in_gO_but_in_RNAAltas += 1
                        ans = {cosine_similarity([vecs[bodypart], vecs[gene]])[0][1]}
                        ans = list(ans)[0]
                        not_in_gO_but_in_RNAAltas.append(ans)
            # 直接进行比对
            print('current B: ', bodypart)
            print('in database : ', count_not_in_gO_but_in_RNAAltas)
            print('not in database : ', count_not_in_gO_RNAAltas)
            # if count_not_in_gO_but_in_hTFtarget != 0:
            #     figName = str(bodypart)
            #     savePath = './LiCO/fig/bodypartSim_hTFtarget/' + figName
            #     calBox(not_in_gO_hTFtarget, not_in_gO_but_in_hTFtarget, savePath, bodypart)
            #     cnt += 1
            #     print(cnt)
            # else:
            #     break

            # 每个B的都标准化处理再拼接，看整体水平
            raw1 = np.array(not_in_gO_RNAAltas).reshape(-1, 1)
            raw2 = np.array(not_in_gO_but_in_RNAAltas).reshape(-1, 1)

            if len(raw2) != 0:
                # 先匹配成标准化的格式，做标准化，再转回去相加
                raw1 = np.array(raw1).reshape(1, -1)
                raw1 = raw1[0]
                raw1 = raw1.tolist()
                raw_totalB_not_in += raw1
                # print('current raw_totalB_not_in : ', raw_totalB_not_in)

                raw2 = np.array(raw2).reshape(1, -1)
                raw2 = raw2[0].tolist()
                raw_totalB_in += raw2

                raw1 = np.array(not_in_gO_RNAAltas).reshape(-1, 1)
                raw2 = np.array(not_in_gO_but_in_RNAAltas).reshape(-1, 1)


                zscore_scaler = preprocessing.StandardScaler()
                z1 = zscore_scaler.fit_transform(raw1)
                z2 = zscore_scaler.fit_transform(raw2)

                # 转换为一行list再相加
                z1 = np.array(z1).reshape(1, -1)
                z1 = z1[0].tolist()
                zscore_totalB_not_in += z1

                z2 = np.array(z2).reshape(1, -1)
                z2 = z2[0].tolist()
                zscore_totalB_in += z2

                minmax_scaler = preprocessing.MinMaxScaler()
                min1 = minmax_scaler.fit_transform(raw1)
                min2 = minmax_scaler.fit_transform(raw2)

                # 转换为一行list再相加
                min1 = np.array(min1).reshape(1, -1)
                min1 = min1[0].tolist()
                minmax_totalB_not_in += min1

                min2 = np.array(min2).reshape(1, -1)
                min2 = min2[0].tolist()
                minmax_totalB_in += min2

                maxabs_scaler = preprocessing.MaxAbsScaler()
                max1 = maxabs_scaler.fit_transform(raw1)
                max2 = maxabs_scaler.fit_transform(raw2)

                # 转换为一行list再相加
                max1 = np.array(max1).reshape(1, -1)
                max1 = max1[0].tolist()
                maxabs_totalB_not_in += max1

                max2 = np.array(max2).reshape(1, -1)
                max2 = max2[0].tolist()
                maxabs_totalB_in += max2

                Robust_scaler = preprocessing.RobustScaler()
                r1 = Robust_scaler.fit_transform(raw1)
                r2 = Robust_scaler.fit_transform(raw2)
                # 转换为一行list再相加
                r1 = np.array(r1).reshape(1, -1)
                r1 = r1[0].tolist()
                Robust_totalB_not_in += r1

                r2 = np.array(r2).reshape(1, -1)
                r2 = r2[0].tolist()
                Robust_totalB_in += r2
            else:
                print('not in if')

    with open('./LiCO/var/' + 'RNAAltas_totalB_test.txt', 'a') as w:
        # t_test
        t, pval = t_test(raw_totalB_not_in, raw_totalB_in)
        w.write('t-test_raw:' + '\t' + str(t) + '\t' + str(pval) + '\n')

        # mannwhitneyu检验
        u, pval = stats.mannwhitneyu(raw_totalB_not_in, raw_totalB_in, alternative='two-sided')
        w.write('mannwhitneyu_raw:' + '\t' + str(t) + '\t' + str(pval) + '\n')

        # t_test
        t, pval = t_test(zscore_totalB_not_in, zscore_totalB_in)
        w.write('t-test_zscore:' + '\t' + str(t) + '\t' + str(pval) + '\n')

        # mannwhitneyu检验
        u, pval = stats.mannwhitneyu(zscore_totalB_not_in, zscore_totalB_in, alternative='two-sided')
        w.write('mannwhitneyu_zscore:' + '\t' + str(t) + '\t' + str(pval) + '\n')

        # t_test
        t, pval = t_test(minmax_totalB_not_in, minmax_totalB_in)
        w.write('t-test_minmax:' + '\t' + str(t) + '\t' + str(pval) + '\n')

        # mannwhitneyu检验
        u, pval = stats.mannwhitneyu(minmax_totalB_not_in, minmax_totalB_in, alternative='two-sided')
        w.write('mannwhitneyu_minmax:' + '\t' + str(t) + '\t' + str(pval) + '\n')

        # t_test
        t, pval = t_test(maxabs_totalB_not_in, maxabs_totalB_in)
        w.write('t-test_maxabs:' + '\t' + str(t) + '\t' + str(pval) + '\n')

        # mannwhitneyu检验
        u, pval = stats.mannwhitneyu(maxabs_totalB_not_in, maxabs_totalB_in, alternative='two-sided')
        w.write('mannwhitneyu_maxabs:' + '\t' + str(t) + '\t' + str(pval) + '\n')

        # t_test
        t, pval = t_test(Robust_totalB_not_in, Robust_totalB_in)
        w.write('t-test_Robust:' + '\t' + str(t) + '\t' + str(pval) + '\n')

        # mannwhitneyu检验
        u, pval = stats.mannwhitneyu(Robust_totalB_not_in, Robust_totalB_in, alternative='two-sided')
        w.write('mannwhitneyu_Robust:' + '\t' + str(t) + '\t' + str(pval) + '\n')
    w.close()

    # 最后统一看箱线图
    labels = 'raw_not', 'raw_in', \
             's_not', 's_in', \
             'min_not', 'min_in', \
             'max_not', 'max1_in', \
             'r_not', 'r_in'

    # print('raw_totalB_not_in : ', raw_totalB_not_in)
    # print('zscore_totalB_not_in : ', zscore_totalB_not_in)
    # print('minmax_totalB_not_in : ', minmax_totalB_not_in)
    # print('maxabs_totalB_not_in : ', maxabs_totalB_not_in)
    # print('Robust_totalB_not_in : ', Robust_totalB_not_in)

    plt.grid(True)  # 显示网格
    plt.boxplot(
        [raw_totalB_not_in, raw_totalB_in,
         zscore_totalB_not_in, zscore_totalB_in,
         minmax_totalB_not_in, minmax_totalB_in,
         maxabs_totalB_not_in, maxabs_totalB_in,
         Robust_totalB_not_in, Robust_totalB_in
         ],
        medianprops={'color': 'red', 'linewidth': '1.2'},
        meanline=True,
        showmeans=True,
        meanprops={'color': 'blue', 'ls': '--', 'linewidth': '1.2'},
        flierprops={"marker": "o", "markerfacecolor": "red", "markersize": 3},
        labels=labels)
    plt.yticks(np.arange(-2, 4.2, 0.2))
    plt.savefig('./LiCO/fig/totalB_sim/' + 'RNAAltas_totalB.png', dpi=450)
    plt.show()
    plt.close()


def cal_box_eachTF_TG():
    vecs = importvec('./LiCO/' + '10_15_7path_addGB_4layerTrain_humanGRN_2_vec.txt')
    geneORGANizer = datas.importgeneItem()
    grn = datas.import_database_grnhg()
    ppi = datas.importppi()

    total_geneset = (set(geneORGANizer['genes'].keys()).union(set(ppi.keys()))).union(set(grn['gene_tfs'].keys()))

    tf_database_hTFtarget_dict = datas.import_database_hTFtarget()['tfs']
    tf_database_TRRUST_dict = datas.import_database_TRRUST()['tfs']

    # gene_database_hTFtarget_dict = datas.import_database_hTFtarget()['genes']
    # gene_database_TRRUST_dict = datas.import_database_TRRUST()['tgs']

    database_TRRUST_dict = datas.import_database_TRRUST()
    database_hTFtarget_dict = datas.import_database_hTFtarget()

    # 根据fmax阈值得到的TF-TG关系
    predict_fmax_tf_tg_dict = datas.import_predict_TF_TG_fmax()

    # 取grn和标准数据库的交集 不能取并集 因为是TF预测TG
    tf_set1 = set(grn['tfs']).intersection(tf_database_hTFtarget_dict.keys())
    tf_set2 = set(grn['tfs']).intersection(tf_database_TRRUST_dict.keys())

    # 新预测的 不在GRN也不在参考数据库的 vs 不在GRN但是在参考数据库中的
    raw_totalTF_not_in = []
    raw_totalTF_in = []
    zscore_totalTF_not_in = []
    zscore_totalTF_in = []
    minmax_totalTF_not_in = []
    minmax_totalTF_in = []
    maxabs_totalTF_not_in = []
    maxabs_totalTF_in = []
    Robust_totalTF_not_in = []
    Robust_totalTF_in = []

    print('len of tf predict dict : ', len(predict_fmax_tf_tg_dict['tfs']))
    for tf in predict_fmax_tf_tg_dict['tfs']:
        # 如果该tf在GRN和hTFtarget的TF并集中(set1)、TRRUST(set2)
        if tf in tf_set1:
            # 遍历预测出的tg，挑出不在GRN且不在参考数据集 和 不在GRN但在参考数据集中的
            count_not_in_grn_hTFtarget = 0
            count_not_in_grn_but_in_hTFtarget = 0
            not_in_grn_hTFtarget = []
            not_in_grn_but_in_hTFtarget = []
            for tg in predict_fmax_tf_tg_dict['tfs'][tf]:
                if tg in vecs:
                    if tg not in grn['tfs'][tf] and tg not in database_hTFtarget_dict['tfs'][tf]:
                        count_not_in_grn_hTFtarget += 1
                        ans = {cosine_similarity([vecs[tf], vecs[tg]])[0][1]}
                        ans = list(ans)[0]
                        not_in_grn_hTFtarget.append(ans)
                    if tg not in grn['tfs'][tf] and tg in database_hTFtarget_dict['tfs'][tf]:
                        count_not_in_grn_but_in_hTFtarget += 1
                        ans = {cosine_similarity([vecs[tf], vecs[tg]])[0][1]}
                        ans = list(ans)[0]
                        not_in_grn_but_in_hTFtarget.append(ans)
            # 直接进行比对
            print('current TF: ', tf)
            print('in database : ', count_not_in_grn_but_in_hTFtarget)
            print('not in database : ', count_not_in_grn_hTFtarget)

            # 每个TF的都标准化处理再拼接，看整体水平 不在 VS 在
            raw1 = np.array(not_in_grn_hTFtarget).reshape(-1, 1)
            raw2 = np.array(not_in_grn_but_in_hTFtarget).reshape(-1, 1)
            # print('raw2: ', raw2)
            # print('count_not_in_grn_but_in_TRRUST: ', count_not_in_grn_but_in_TRRUST)

            if count_not_in_grn_but_in_hTFtarget != 0:   # len(raw2) != 0不能判定
                print('in if ')
                # print('count_not_in_grn_but_in_TRRUST : ', count_not_in_grn_but_in_TRRUST)
                # 先匹配成标准化的格式，做标准化，再转回去相加
                raw1 = np.array(raw1).reshape(1, -1)
                raw1 = raw1[0]
                raw1 = raw1.tolist()
                raw_totalTF_not_in += raw1
                # print('current raw_totalB_not_in : ', raw_totalB_not_in)

                raw2 = np.array(raw2).reshape(1, -1)
                raw2 = raw2[0].tolist()
                raw_totalTF_in += raw2

                raw1 = np.array(not_in_grn_hTFtarget).reshape(-1, 1)
                raw2 = np.array(not_in_grn_but_in_hTFtarget).reshape(-1, 1)

                zscore_scaler = preprocessing.StandardScaler()
                z1 = zscore_scaler.fit_transform(raw1)
                z2 = zscore_scaler.fit_transform(raw2)

                # 转换为一行list再相加
                z1 = np.array(z1).reshape(1, -1)
                z1 = z1[0].tolist()
                zscore_totalTF_not_in += z1

                z2 = np.array(z2).reshape(1, -1)
                z2 = z2[0].tolist()
                zscore_totalTF_in += z2

                minmax_scaler = preprocessing.MinMaxScaler()
                min1 = minmax_scaler.fit_transform(raw1)
                min2 = minmax_scaler.fit_transform(raw2)

                # 转换为一行list再相加
                min1 = np.array(min1).reshape(1, -1)
                min1 = min1[0].tolist()
                minmax_totalTF_not_in += min1

                min2 = np.array(min2).reshape(1, -1)
                min2 = min2[0].tolist()
                minmax_totalTF_in += min2

                maxabs_scaler = preprocessing.MaxAbsScaler()
                max1 = maxabs_scaler.fit_transform(raw1)
                max2 = maxabs_scaler.fit_transform(raw2)

                # 转换为一行list再相加
                max1 = np.array(max1).reshape(1, -1)
                max1 = max1[0].tolist()
                maxabs_totalTF_not_in += max1

                max2 = np.array(max2).reshape(1, -1)
                max2 = max2[0].tolist()
                maxabs_totalTF_in += max2

                Robust_scaler = preprocessing.RobustScaler()
                r1 = Robust_scaler.fit_transform(raw1)
                r2 = Robust_scaler.fit_transform(raw2)
                # 转换为一行list再相加
                r1 = np.array(r1).reshape(1, -1)
                r1 = r1[0].tolist()
                Robust_totalTF_not_in += r1

                r2 = np.array(r2).reshape(1, -1)
                r2 = r2[0].tolist()
                Robust_totalTF_in += r2
            else:
                print('not in if, break')
    if len(raw2) != 0:
        with open('./LiCO/var/' + 'hTFtarget_totalTF_test.txt', 'a') as w:
            print('open txt')
            # t_test
            t, pval = t_test(raw_totalTF_not_in, raw_totalTF_in)
            w.write('t-test_raw:' + '\t' + str(t) + '\t' + str(pval) + '\n')

            # mannwhitneyu检验
            u, pval = stats.mannwhitneyu(raw_totalTF_not_in, raw_totalTF_in, alternative='two-sided')
            w.write('mannwhitneyu_raw:' + '\t' + str(t) + '\t' + str(pval) + '\n')

            # t_test
            t, pval = t_test(zscore_totalTF_not_in, zscore_totalTF_in)
            w.write('t-test_zscore:' + '\t' + str(t) + '\t' + str(pval) + '\n')

            # mannwhitneyu检验
            u, pval = stats.mannwhitneyu(zscore_totalTF_not_in, zscore_totalTF_in, alternative='two-sided')
            w.write('mannwhitneyu_zscore:' + '\t' + str(t) + '\t' + str(pval) + '\n')

            # t_test
            t, pval = t_test(minmax_totalTF_not_in, minmax_totalTF_in)
            w.write('t-test_minmax:' + '\t' + str(t) + '\t' + str(pval) + '\n')

            # mannwhitneyu检验
            u, pval = stats.mannwhitneyu(minmax_totalTF_not_in, minmax_totalTF_in, alternative='two-sided')
            w.write('mannwhitneyu_minmax:' + '\t' + str(t) + '\t' + str(pval) + '\n')

            # t_test
            t, pval = t_test(maxabs_totalTF_not_in, maxabs_totalTF_in)
            w.write('t-test_maxabs:' + '\t' + str(t) + '\t' + str(pval) + '\n')

            # mannwhitneyu检验
            u, pval = stats.mannwhitneyu(maxabs_totalTF_not_in, maxabs_totalTF_in, alternative='two-sided')
            w.write('mannwhitneyu_maxabs:' + '\t' + str(t) + '\t' + str(pval) + '\n')

            # t_test
            t, pval = t_test(Robust_totalTF_not_in, Robust_totalTF_in)
            w.write('t-test_Robust:' + '\t' + str(t) + '\t' + str(pval) + '\n')

            # mannwhitneyu检验
            u, pval = stats.mannwhitneyu(Robust_totalTF_not_in, Robust_totalTF_in, alternative='two-sided')
            w.write('mannwhitneyu_Robust:' + '\t' + str(t) + '\t' + str(pval) + '\n')
        w.close()

        # 最后统一看箱线图
        labels = 'raw_not', 'raw_in', \
                 'z_not', 'z_in', \
                 'min_not', 'min_in', \
                 'max_not', 'max1_in', \
                 'r_not', 'r_in'

        # print('raw_totalB_not_in : ', raw_totalB_not_in)
        # print('zscore_totalB_not_in : ', zscore_totalB_not_in)
        # print('minmax_totalB_not_in : ', minmax_totalB_not_in)
        # print('maxabs_totalB_not_in : ', maxabs_totalB_not_in)
        # print('Robust_totalB_not_in : ', Robust_totalB_not_in)

        plt.grid(True)  # 显示网格
        plt.boxplot(
            [raw_totalTF_not_in, raw_totalTF_in,
             zscore_totalTF_not_in, zscore_totalTF_in,
             minmax_totalTF_not_in, minmax_totalTF_in,
             maxabs_totalTF_not_in, maxabs_totalTF_in,
             Robust_totalTF_not_in, Robust_totalTF_in
             ],
            medianprops={'color': 'red', 'linewidth': '1.2'},
            meanline=True,
            showmeans=True,
            meanprops={'color': 'blue', 'ls': '--', 'linewidth': '1.2'},
            flierprops={"marker": "o", "markerfacecolor": "red", "markersize": 3},
            labels=labels)
        plt.yticks(np.arange(-3.0, 15.5, 0.5))
        plt.savefig('./LiCO/fig/totalTF_sim/' + 'hTFtarget_totalTF.png', dpi=450)
        plt.show()
        plt.close()



def cal_box_b_re():
    vecs = importvec('./LiCO/' + '10_15_7path_addGB_4layerTrain_humanGRN_2_vec.txt')
    geneORGANizer = datas.importgeneItem()
    grn = datas.import_database_grnhg()
    ppi = datas.importppi()

    total_geneset = (set(geneORGANizer['genes'].keys()).union(set(ppi.keys()))).union(set(grn['gene_tfs'].keys()))


    # 参考数据库GTEx
    GTEx = datas.import_database_GTEx()

    # 根据fmax阈值得到的re-b关系
    predict_fmax_re_g_dict = datas.import_predict_b_g_fmax()


    # Boxplot test
    # 一、三个数据库自身的分布
    # geneORGANizer和3个数据库的B-G关系的分布
    gO_B_G = []
    for bodypart in geneORGANizer['bodyparts']:
        if bodypart in vecs:
            for gene in geneORGANizer['bodyparts'][bodypart]:
                if gene in vecs:
                    ans = {cosine_similarity([vecs[bodypart], vecs[gene]])[0][1]}
                    ans = list(ans)[0]
                    gO_B_G.append(ans)

    RNAAltas_B_G = []
    for bodypart in bodypart_database_RNAAltas_dict:
        if bodypart in vecs:
            for gene in bodypart_database_RNAAltas_dict[bodypart]:
                if gene in vecs:
                    ans = {cosine_similarity([vecs[bodypart], vecs[gene]])[0][1]}
                    ans = list(ans)[0]
                    RNAAltas_B_G.append(ans)

    hTFtarget_B_G = []
    for bodypart in bodypart_database_hTFtarget_dict:
        if bodypart in vecs:
            for gene in bodypart_database_hTFtarget_dict[bodypart]:
                if gene in vecs:
                    ans = {cosine_similarity([vecs[bodypart], vecs[gene]])[0][1]}
                    ans = list(ans)[0]
                    hTFtarget_B_G.append(ans)

    # 二、新预测的数据集
    # 新预测的 不在GRN也不在参考数据库的 vs 不在GRN但是在参考数据库中的
    # count_not_in_gO_RNAAltas = 0
    # count_not_in_gO_but_in_RNAAltas = 0
    #
    # not_in_gO_RNAAltas = []
    # not_in_gO_but_in_RNAAltas = []
    #
    # for bodypart in predict_fmax_b_g_dict['bodyparts']:
    #     # bodypart必在vecs中，这里要保证在参考数据库中
    #     if bodypart in bodypart_database_RNAAltas_dict:
    #         # 遍历预测出的tg，挑出不在GRN且不在参考数据集 和 不在GRN但在参考数据集中的
    #         for gene in predict_fmax_b_g_dict['bodyparts'][bodypart]:
    #             if gene in vecs:
    #                 if gene not in geneORGANizer['bodyparts'][bodypart] and gene not in bodypart_database_RNAAltas_dict[bodypart]:
    #                     count_not_in_gO_RNAAltas += 1
    #                     ans = {cosine_similarity([vecs[bodypart], vecs[gene]])[0][1]}
    #                     ans = list(ans)[0]
    #                     not_in_gO_RNAAltas.append(ans)
    #                 if gene not in geneORGANizer['bodyparts'][bodypart] and gene in bodypart_database_RNAAltas_dict[bodypart]:
    #                     count_not_in_gO_but_in_RNAAltas += 1
    #                     ans = {cosine_similarity([vecs[bodypart], vecs[gene]])[0][1]}
    #                     ans = list(ans)[0]
    #                     not_in_gO_but_in_RNAAltas.append(ans)
    #
    # print('in database : ', count_not_in_gO_but_in_RNAAltas)
    # print('not in database : ', count_not_in_gO_RNAAltas)



    count_not_in_gO_hTFtarget = 0
    count_not_in_gO_but_in_hTFtarget = 0

    not_in_gO_hTFtarget = []
    not_in_gO_but_in_hTFtarget = []

    for bodypart in predict_fmax_b_g_dict['bodyparts']:
        if bodypart in bodypart_database_hTFtarget_dict:
            # 遍历预测出的tg，挑出不在GRN且不在参考数据集 和 不在GRN但在参考数据集中的
            for gene in predict_fmax_b_g_dict['bodyparts'][bodypart]:
                if gene in vecs:
                    if gene not in geneORGANizer['bodyparts'][bodypart] and gene not in bodypart_database_hTFtarget_dict[bodypart]:
                        count_not_in_gO_hTFtarget += 1
                        ans = {cosine_similarity([vecs[bodypart], vecs[gene]])[0][1]}
                        ans = list(ans)[0]
                        not_in_gO_hTFtarget.append(ans)
                    if gene not in geneORGANizer['bodyparts'][bodypart] and gene in bodypart_database_hTFtarget_dict[bodypart]:
                        count_not_in_gO_but_in_hTFtarget += 1
                        ans = {cosine_similarity([vecs[bodypart], vecs[gene]])[0][1]}
                        ans = list(ans)[0]
                        not_in_gO_but_in_hTFtarget.append(ans)

    print('in database : ', count_not_in_gO_but_in_hTFtarget)
    print('not in database : ', count_not_in_gO_hTFtarget)

    # 做标准化
    # z-score
    zscore_scaler = preprocessing.StandardScaler()
    not_in_gO_RNAAltas = np.array(not_in_gO_hTFtarget).reshape(-1, 1)  # 1规定列数为1，-1表根据给定的列数自动分配行数
    not_in_gO_but_in_RNAAltas = np.array(not_in_gO_but_in_hTFtarget).reshape(-1, 1)

    data_score_1 = zscore_scaler.fit_transform(not_in_gO_RNAAltas)
    data_score_2 = zscore_scaler.fit_transform(not_in_gO_but_in_RNAAltas)

    # print('original: ', not_in_grn_TFtarget)
    # print('transformed: ', data_score_1)

    # # max-min
    minmax_scaler = preprocessing.MinMaxScaler()
    data_minmax_1 = minmax_scaler.fit_transform(not_in_gO_RNAAltas)
    data_minmax_2 = minmax_scaler.fit_transform(not_in_gO_but_in_RNAAltas)

    # maxAbs
    maxabs_scaler = preprocessing.MaxAbsScaler()
    data_maxabs_1 = maxabs_scaler.fit_transform(not_in_gO_RNAAltas)
    data_maxabs_2 = maxabs_scaler.fit_transform(not_in_gO_but_in_RNAAltas)

    # robustScaler
    Robust_scaler = preprocessing.RobustScaler()
    data_Robust_1 = Robust_scaler.fit_transform(not_in_gO_RNAAltas)
    data_Robust_2 = Robust_scaler.fit_transform(not_in_gO_but_in_RNAAltas)

    # Normalizer
    Normal_l1_scaler = preprocessing.Normalizer(norm='l1')
    data_Normal_l1_1 = Normal_l1_scaler.fit_transform(not_in_gO_RNAAltas)
    data_Normal_l1_2 = Normal_l1_scaler.fit_transform(not_in_gO_but_in_RNAAltas)

    Normal_l2_scaler = preprocessing.Normalizer(norm='l2')
    data_Normal_l2_1 = Normal_l2_scaler.fit_transform(not_in_gO_RNAAltas)
    data_Normal_l2_2 = Normal_l2_scaler.fit_transform(not_in_gO_but_in_RNAAltas)

    Normal_max_scaler = preprocessing.Normalizer(norm='max')
    data_Normal_max_1 = Normal_max_scaler.fit_transform(not_in_gO_RNAAltas)
    data_Normal_max_2 = Normal_max_scaler.fit_transform(not_in_gO_but_in_RNAAltas)

    # print('count_one : ', count_one)
    # print('count_zero : ', count_zero)
    # 绘制箱线图
    labels = 'original_1', 'original_2', 'zscore_1', 'zscore_2', 'minmax_1', 'minmax_2', 'maxabs_1', 'maxabs_2', 'Robust_1', 'Robust_2', 'L1_1', 'L1_2', 'L2_1', 'L2_2', 'Normal_max_1', 'Normal_max_2'
    plt.grid(True)  # 显示网格
    plt.boxplot([not_in_gO_RNAAltas, not_in_gO_but_in_RNAAltas, data_score_1, data_score_2, data_minmax_1, data_minmax_2, data_maxabs_1, data_maxabs_2, data_Robust_1,
                 data_Robust_2, data_Normal_l1_1, data_Normal_l1_2, data_Normal_l2_1, data_Normal_l2_2, data_Normal_max_1, data_Normal_max_2],
                medianprops={'color': 'red', 'linewidth': '1.5'},
                meanline=True,
                showmeans=True,
                meanprops={'color': 'blue', 'ls': '--', 'linewidth': '1.5'},
                flierprops={"marker": "o", "markerfacecolor": "red", "markersize": 5},
                labels=labels)
    plt.yticks(np.arange(-6, 4.5, 0.5))
    plt.show()
    plt.savefig('./LiCO/fig/box_preprocess_hTFtarget_not_in_vs_in.png')


def calBox(dict1, dict2, savePath, bodypart):
    # 返回值：2种情况下各自有5个返回值
    with open('./LiCO/var/' + 'hTFtarget_ttest.txt', 'a') as w:


        # 做标准化
        # z-score
        zscore_scaler = preprocessing.StandardScaler()
        # d1 = np.array(relatedRE_relatedTFs_dict).reshape(-1, 1)  # 1规定列数为1，-1表根据给定的列数自动分配行数
        d2 = np.array(dict1).reshape(-1, 1)
        # d3 = np.array(unrelatedRE_relatedTFs_dict).reshape(-1, 1)
        d4 = np.array(dict2).reshape(-1, 1)

        # t_test
        t, pval = t_test(d2, d4)
        w.write(str(bodypart) + '\t' + 't-test_raw:' + '\t' + str(t) + '\t' + str(pval) + '\n')


        # mannwhitneyu检验
        u, pval = stats.mannwhitneyu(d2, d4, alternative='two-sided')
        w.write(str(bodypart) + '\t' + 'mannwhitneyu_raw:' + '\t' + str(t) + '\t' + str(pval) + '\n')


        # s1 = zscore_scaler.fit_transform(d1)
        s2 = zscore_scaler.fit_transform(d2)
        # s3 = zscore_scaler.fit_transform(d3)
        s4 = zscore_scaler.fit_transform(d4)

        t, pval = t_test(s2, s4)
        w.write(str(bodypart) + '\t' + 't-test_zscore:' + '\t' + str(t) + '\t' + str(pval) + '\n')

        # mannwhitneyu检验
        u, pval = stats.mannwhitneyu(s2, s4, alternative='two-sided')
        w.write(str(bodypart) + '\t' + 'mannwhitneyu_zscore:' + '\t' + str(t) + '\t' + str(pval) + '\n')


        # # max-min
        minmax_scaler = preprocessing.MinMaxScaler()
        # min1 = minmax_scaler.fit_transform(d1)
        min2 = minmax_scaler.fit_transform(d2)
        # min3 = minmax_scaler.fit_transform(d3)
        min4 = minmax_scaler.fit_transform(d4)

        t, pval = t_test(min2, min4)
        w.write(str(bodypart) + '\t' + 't-test_max_min:' + '\t' + str(t) + '\t' + str(pval) + '\n')

        # mannwhitneyu检验
        u, pval = stats.mannwhitneyu(min2, min4, alternative='two-sided')
        w.write(str(bodypart) + '\t' + 'mannwhitneyu_max_min:' + '\t' + str(t) + '\t' + str(pval) + '\n')

        # maxAbs
        maxabs_scaler = preprocessing.MaxAbsScaler()
        # max1 = maxabs_scaler.fit_transform(d1)
        max2 = maxabs_scaler.fit_transform(d2)
        # max3 = maxabs_scaler.fit_transform(d3)
        max4 = maxabs_scaler.fit_transform(d4)

        t, pval = t_test(max2, max4)
        w.write(str(bodypart) + '\t' + 't-test_maxabs:' + '\t' + str(t) + '\t' + str(pval) + '\n')

        # mannwhitneyu检验
        u, pval = stats.mannwhitneyu(max2, max4, alternative='two-sided')
        w.write(str(bodypart) + '\t' + 'mannwhitneyu_maxabs:' + '\t' + str(t) + '\t' + str(pval) + '\n')


        # robustScaler
        Robust_scaler = preprocessing.RobustScaler()
        # r1 = Robust_scaler.fit_transform(d1)
        r2 = Robust_scaler.fit_transform(d2)
        # r3 = Robust_scaler.fit_transform(d3)
        r4 = Robust_scaler.fit_transform(d4)

        t, pval = t_test(r2, r4)
        w.write(str(bodypart) + '\t' + 't-test_Robust_scaler:' + '\t' + str(t) + '\t' + str(pval) + '\n')

        # mannwhitneyu检验
        u, pval = stats.mannwhitneyu(r2, r4, alternative='two-sided')
        w.write(str(bodypart) + '\t' + 'mannwhitneyu_Robust_scaler:' + '\t' + str(t) + '\t' + str(pval) + '\n')

        # # Normalizer 显示都是一条线
        # Normal_l1_scaler = preprocessing.Normalizer(norm='l1')
        # l1_1 = Normal_l1_scaler.fit_transform(d1)
        # l1_2 = Normal_l1_scaler.fit_transform(d2)
        # l1_3 = Normal_l1_scaler.fit_transform(d3)
        # l1_4 = Normal_l1_scaler.fit_transform(d4)
        #
        # Normal_l2_scaler = preprocessing.Normalizer(norm='l2')
        # l2_1 = Normal_l2_scaler.fit_transform(d1)
        # l2_2 = Normal_l2_scaler.fit_transform(d2)
        # l2_3 = Normal_l2_scaler.fit_transform(d3)
        # l2_4 = Normal_l2_scaler.fit_transform(d4)
        #
        # Normal_max_scaler = preprocessing.Normalizer(norm='max')
        # Nmax_1 = Normal_max_scaler.fit_transform(d1)
        # Nmax_2 = Normal_max_scaler.fit_transform(d2)
        # Nmax_3 = Normal_max_scaler.fit_transform(d3)
        # Nmax_4 = Normal_max_scaler.fit_transform(d4)

        # print('count_one : ', count_one)
        # print('count_zero : ', count_zero)
        # 绘制箱线图
        # labels = 'org1', 'org2', 'org3', 'org4', \
        #          's1', 's2', 's3', 's4', \
        #          'min1', 'min2', 'min3', 'min4', \
        #          'max1', 'max12', 'max3', 'max4', \
        #          'r1', 'r2', 'r3', 'r4'
        labels = 'raw1', 'raw2', \
                 's1', 's2', \
                 'min1', 'min2', \
                 'max1', 'max12', \
                 'r1', 'r2'

        plt.grid(True)  # 显示网格
        plt.boxplot(
            [d2, d4,
             s2, s4,
             min2, min4,
             max2, max4,
             r2, r4
             ],
            medianprops={'color': 'red', 'linewidth': '1.2'},
            meanline=True,
            showmeans=True,
            meanprops={'color': 'blue', 'ls': '--', 'linewidth': '1.2'},
            flierprops={"marker": "o", "markerfacecolor": "red", "markersize": 3},
            labels=labels)
        plt.yticks(np.arange(-2, 4.2, 0.2))
        plt.savefig(savePath + ".png", dpi=459)
        # plt.show()
        plt.close()
    w.close()


def t_test(e, c, equal_value=False):
    t, pval = scipy.stats.ttest_ind(e, c, equal_var=equal_value)
    return t, pval

def test():
    not_in_grn_TRRUST = [1, 2, 3, 4, 5]
    not_in_grn_but_in_TRRUST = [2, 4, 6, 8, 10]


    not_in_grn_TRRUST = np.array(not_in_grn_TRRUST).reshape((-1, 1))
    not_in_grn_TRRUST = np.array(not_in_grn_TRRUST).reshape((1, -1))

    not_in_grn_but_in_TRRUST = np.array(not_in_grn_but_in_TRRUST).reshape((-1, 1))
    not_in_grn_but_in_TRRUST = np.array(not_in_grn_but_in_TRRUST).reshape((1, -1))

    not_in_grn_TRRUST = not_in_grn_TRRUST.tolist()
    not_in_grn_but_in_TRRUST = not_in_grn_but_in_TRRUST.tolist()

    # print(not_in_grn_TRRUST[0])
    # print(not_in_grn_but_in_TRRUST[0])
    #
    # # 方差齐性检验
    # stat, p = levene(not_in_grn_TRRUST[0], not_in_grn_but_in_TRRUST[0])
    # print('raw_levene:')
    # print(stat, p)
    #
    # # 方差无显著差异
    # if p >= 0.05:
    #     t, pval = t_test(not_in_grn_TRRUST[0], not_in_grn_but_in_TRRUST[0])
    #     print('raw:')
    #     print(t, pval)
    # # 方差有显著差异
    # else:
    #     print('false')
    #     t, pval = t_test(not_in_grn_TRRUST[0], not_in_grn_but_in_TRRUST[0], False)
    #     print('raw:')
    #     print(t, pval)




if __name__ == '__main__':
    # test()
    # lzm_calTopSim()
    # main()
    # score_tf_predict_tg()
    # score_gene_predict_bodypart()
    # diff_geneORGNizer_cellMarker()
    score_bodypart_predict_gene()
    # score_tg_predict_tf()
    # diff_geneORGNizer_cellMarker()
    # # calscore()
    # cal_box_tftg() # 有逻辑问题，需要修改 11/3/21:37  已修改11/4/11:30
    # cal_box_g_b()
    # test()
    # cal_box_eachB_G()
    # cal_box_eachTF_TG()