from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import importdatas as datas
import pickle
import json

path = './datamgi/'
notename = '-noDis'
filename = '/share/home/liangzhongming/RE_ORGANize_4layers/RE-ORGANizer-master/generateREGOA/'
# 计算每个term的thres
def testterm(prob, stand):
    threshigh, thresfmax, threslow = 1, 1, 1
    fmax = 0
    findhigh = False
    findlow = False
    p = sum(stand)
    print('total true = ', p)
    if p == 0:
        return 1, 1, 1
    # 计算rec、pre及f1
    for t in range(1, 1000):  # 阈值范围[1/1000, 1)
        thres = t/1000
        tp = 0
        fp = 0
        for x in range(len(prob)):
            if prob[x] >= thres:
                if stand[x] == 1:
                    tp += 1
                else:
                    fp += 1
        rec = tp / p
        # print('now rec = ', rec)
        if tp+fp != 0:
            pre = tp/(tp + fp)
            # print(f"pre: {pre}")
            # print('now pre = ', pre)
            if pre >= 0.3 and not findlow:
                findlow = True
                threslow = thres
            if pre >= 0.7 and not findhigh:  #  只找满足条件的第一个 解释了三个值都是0.001的情况
                findhigh = True
                threshigh = thres
            if pre + rec != 0:
                f1 = 2*pre*rec/(pre+rec)
                if f1 > fmax:
                    fmax = f1
                    thresfmax = thres
                    # print('now fmax = ', fmax)
                    # print('and now thres = ', thres)
    return threshigh, thresfmax, threslow

# 构造prob和stand 但似乎我直接用已有的正负样本文件构建即可
# def cal(function, goa, go, vecs):
    global functions
    # goterm = list(goa['terms'].keys())
    gOorgan = list(gO['organs'].keys())
    prob = dict()
    standard = dict()

    proteins = set()
    # 将存在向量表示的'东西'，存入proteins，同时在prob和standard字典中创建以其为key的字典
    with open('./data/validpro.txt', 'r') as f:  # 这又是个什么文件？
        count = 0
        for line in f:
            line = line.split()[0]
            if line in vecs.keys():
                prob[line] = dict()
                standard[line] = dict()
                proteins.add(line)
            count += 1


    M = len(prob)
    count = 0
    for protein in prob:
        for organ in gOorgan:  # 按下边构建负样本的操作，该term应该是不与gene有关联的
            # 对prob中的每个'东西'，与term计算余弦相似度
            prob[protein][organ] = cosine_similarity([vecs[organ], vecs[protein]])[0][1]
            # 标记为0，表示为负样本  ？
            standard[protein][organ] = 0
        terms = set(go.keys())
        while terms:
            term = terms.pop()
            nowmax = prob[protein][term]
            nowterm = term
            for tterm in go[term]['descent'] & set(prob[protein].keys()):
                if prob[protein][tterm] > nowmax:
                    nowmax = prob[protein][tterm]
                    nowterm = tterm
            prob[protein][term] = nowmax
            for tterm in go[term]['descent'] & go[nowterm]['ancesent'] & terms:
                prob[protein][tterm] = nowmax
            terms = terms - go[term]['descent'] & go[nowterm]['ancesent']
        count += 1
        print(count, '/', M, ' test')

    with open('./data/valid.txt', 'r') as f:
        count = 0
        for line in f:
            line = line.split()
            if line[0] in prob and line[1] in goterm:
                # 正样本
                standard[line[0]][line[1]] = 1
                for term in set(goterm) & go[line[1]]['ancesent']:
                    # 构建正样本
                    standard[line[0]][term] = 1
            count += 1
            print(count, '/goa test')

    # 输出每个term的threshold
    with open('./data/'+function+'threshold.txt', 'w') as f:
        f.write('go_id\t pre=0.7\t Fmax\t pre=0.3\n')
        count = 0
        M = len(goterm)
        for term in goterm:
            threshigh ,thresfmax,threslow = testterm([prob[protein][term] for protein in proteins], [standard[protein][term] for protein in proteins])
            f.write(term+'\t'+str(threshigh)+'\t'+str(thresfmax)+'\t'+str(threslow)+'\n')
            count += 1
            print(count, '/', M, 'term test')

# 根据正负样本生成stand和score
# def lzm_cal(g_O, vecs):
#     print('run lzm_cal()')
#     g_O = datas.importgene_organs('./New_gene_organs_2.txt')
#     prob = dict()
#     stand = dict()
#     geneSet = set()
#     with open(path + './pos_neg/pos.txt', 'r') as f:
#         print('open pos.txt')
#         count = 0
#         for line in f:
#             line = line.split()
#             gene = line[0]
#             organ = line[1]
#             for index in range(2, len(line)):  # organs名字可能是'xxx xxx xxx'形式
#                 if line[index].isalpha():
#                     organ += ' '
#                     organ += line[index]
#                 else:
#                     break

#             if line[0] in vecs.keys() and organ in vecs.keys() & g_O['organs'].keys():  # 如果gene存在vecs文件中，且organ存在vecs文件和g_O文件中
#                 if organ not in vecs.keys():
#                     print(organ, 'not in vecs.keys()')
#                 if gene not in geneSet:
#                     geneSet.add(gene)
#                 if gene not in prob:
#                     prob[gene] = dict()
#                 prob[gene][organ] = cosine_similarity([vecs[organ], vecs[gene]])[0][1]
#                 # if organ == 'outer ear':
#                 #     print('cur prob[gene][outer ear] :', prob[gene][organ])
#                 if gene not in stand:
#                     stand[gene] = dict()
#                 stand[gene][organ] = 1
#                 # if organ == 'outer ear':
#                 #     print('cur stand[gene][outer ear]: ', stand[gene][organ])
#                 count += 1
#                 if count % 1000 == 0:
#                     print(count, 'in 120171 pos')
#             # # 用于测试，减少耗时
#             # if count == 12000:
#             #     break

#     with open(path + './pos_neg/neg.txt', 'r') as f:
#         print('open neg.txt')
#         count = 0
#         for line in f:
#             line = line.split()
#             gene = line[0]
#             organ = line[1]
#             for index in range(2, len(line)):  # organs名字可能是'xxx xxx xxx'形式
#                 if line[index].isalpha():
#                     organ += ' '
#                     organ += line[index]
#                 else:
#                     break


#             if line[0] in vecs.keys() and organ in vecs.keys() & g_O['organs'].keys():  # 如果gene存在vecs文件中，且organ存在vecs文件和g_O文件中
#                 if gene not in prob:
#                     prob[gene] = dict()
#                 prob[gene][organ] = cosine_similarity([vecs[organ], vecs[gene]])[0][1]
#                 # print('cur prob[gene][organ] :', prob[gene][organ])
#                 if gene not in stand:
#                     stand[gene] = dict()
#                 stand[gene][organ] = 0
#                 # print('cur stand[gene][organ]: ', stand[gene][organ])
#                 count += 1
#                 if count % 1000 == 0:
#                     print(count, 'in 312591 neg')
#             # # 用于测试，减少耗时
#             # if count == 12000:
#             #     break

#     # 测试输出形式是否正确
#     # for organ in g_O['organs'].keys():
#         # for gene in geneSet:
#         #     print('organ : ', organ)
#         #     print(' gene : ', gene)
#         #     print(' prob[gene][organ] : ', prob[gene][organ])
#         #     print(' stand[gene][organ] : ', stand[gene][organ])
#         #
#         # print([prob[gene][organ] for gene in geneSet])
#         # print([stand[gene][organ] for gene in geneSet])

#     with open('./datamgi/diffpaths/6metapath_add_OGOG/cal_prob.txt', 'w') as f:
#         f.write(str(prob))
#         f.close()
#     with open('./datamgi/diffpaths/6metapath_add_OGOG/cal_stand.txt', 'w') as f:
#         f.write((str(stand)))
#         f.close()

#     with open('./datamgi/diffpaths/6metapath_add_OGOG/cal_prob.pkl', 'wb') as f:
#         pickle.dump(prob, f)
#         f.close()
#     with open('./datamgi/diffpaths/6metapath_add_OGOG/cal_stand.pkl', 'wb') as f:
#         pickle.dump(stand, f)
#         f.close()

#     geneSet = set()
#     with open(path + './pos_neg/pos.txt', 'r') as f:
#         for line in f:
#             line = line.split()
#             gene = line[0]
#             if gene not in geneSet:
#                 if gene in vecs.keys():
#                     geneSet.add(gene)

#     with open('./datamgi/diffpaths/6metapath_add_OGOG/cal_prob.pkl', 'rb') as f:
#         prob = pickle.load(f)
#         f.close()
#     with open('./datamgi/diffpaths/6metapath_add_OGOG/cal_stand.pkl', 'rb') as f:
#         stand = pickle.load(f)
#         f.close()




#     # 输出每个organ的threshold
#     with open('./datamgi/diffpaths/6metapath_add_OGOG/' + 'cal_organs_' + 'threshold.txt', 'w') as f:
#         f.write('organ\t pre=0.7\t Fmax\t pre=0.3\n')
#         count = 0
#         M = len(g_O['organs'].keys())
#         for organ in g_O['organs'].keys():
#             # 传入testterm的是一个organ的所有gene的stand和prob，是dict()形式
#             # for gene in geneSet:
#             #     threshigh, thresfmax, threslow = testterm(prob[gene][organ], stand[gene][organ])
#             #     f.write(organ + '\t' + str(threshigh) + '\t' + str(thresfmax) + '\t' + str(threslow) + '\n')
#             #     count += 1
#             #     print(count, '/', M, 'organ test')
#             if organ not in stand:
#                 print(organ, 'not in stand')
#             if organ not in prob.keys():
#                 print(organ, 'not in prob')
#             threshigh, thresfmax, threslow = testterm([prob[gene][organ] for gene in geneSet], [stand[gene][organ] for gene in geneSet])
#             f.write(organ+'\t'+str(threshigh)+'\t'+str(thresfmax)+'\t'+str(threslow)+'\n')
#             count += 1
#             print(count, '/', M, 'organ test')

# lzm_cal()的另一种写法
def lzm_cal1(res, gO_B, gO_G, vecs):

    prob = dict()
    stand = dict()
    geneSet = set()
    average = 0
    count = 0

    for bodypart in gO_B.keys():
        if bodypart in vecs:
            if bodypart not in prob.keys():
                prob[bodypart] = dict()
            if bodypart not in stand.keys():
                stand[bodypart] = dict()

            for gene in gO_G.keys():
                if gene in vecs:
                    if gene not in geneSet:
                        geneSet.add(gene)
                    if gene in gO_B[bodypart]:
                        prob[bodypart][gene] = cosine_similarity([vecs[bodypart], vecs[gene]])[0][1]
                        stand[bodypart][gene] = 1
                    elif gene not in gO_B[bodypart]:
                        prob[bodypart][gene] = cosine_similarity([vecs[bodypart], vecs[gene]])[0][1]
                        stand[bodypart][gene] = 0
                    average += prob[bodypart][gene]
                    count += 1

    print('average cossim = ', average/count)
    # 输出每个organ的threshold
    # with open('./LiCO/threshold/' + 're-g-b' + 'threshold.txt', 'w+') as f:
    with open('/share/home/liangzhongming/NetOrganizer/RE-ORGANizer-master/generateREGOA/gwas_result/test-threshold.txt', 'w') as f:
        f.write('organ\t pre=0.7\t Fmax\t pre=0.3\n')
        count = 0
        M = len(gO_B.keys())
        for bodypart in gO_B.keys():

            # 传入testterm的是一个organ的所有gene的stand和prob，是dict()形式
            # for gene in geneSet:
            #     threshigh, thresfmax, threslow = testterm(prob[gene][organ], stand[gene][organ])
            #     f.write(organ + '\t' + str(threshigh) + '\t' + str(thresfmax) + '\t' + str(threslow) + '\n')
            #     count += 1
            #     print(count, '/', M, 'organ test')
            # in_prob = dict()
            # in_stand = dict()
            # for gene in geneSet:
            #     print('gene is:', gene, ' organ is:', organ, prob[gene][organ], '\t')
            #     in_prob[organ][gene] = prob[gene][organ]
            #     in_stand[organ][gene] = stand[gene][organ]
            #     print('in_stand[organ]:', in_stand[organ])

            threshigh, thresfmax, threslow = testterm([prob[bodypart][gene] for gene in geneSet], [stand[bodypart][gene] for gene in geneSet])
            f.write(bodypart+'\t'+str(threshigh)+'\t'+str(thresfmax)+'\t'+str(threslow)+'\n')
            count += 1
            print(count, '/', M, 'gO_B')


# lzm_cal()的另一种写法
def lzm_cal_re_g_b(res, gO_B, gO_G, vecs):
    grn = datas.import_database_grnhg()
    geneORGANizer = datas.import_database_geneORGANizer()
    ppi = datas.importppi()
    total_geneset = (set(geneORGANizer['genes'].keys()).union(set(ppi.keys()))).union(set(grn['gene_tfs'].keys()))

    prob = dict()
    stand = dict()
    reSet = set()
    bodypartSet = set()
    tissueSet = set()
    average = 0
    cnt_vecgene = 0
    genes = set()

    # for gene in total_geneset:
    # for gene in grn['gene_tfs']:
    for gene in geneORGANizer['genes']:
        if gene in vecs:
            prob[gene] = dict()
            stand[gene] = dict()
            genes.add(gene)
            cnt_vecgene += 1
    
    print(f"cnt_vecgene :{cnt_vecgene}")

    cnt_set0 = 0
    cnt_set1 = 0
    cnt_gene = 0
    M = len(prob)
    for gene in prob:
        cnt_gene += 1
        for organ in geneORGANizer['bodyparts']:
            prob[gene][organ] = cosine_similarity([vecs[organ] , vecs[gene]])[0][1]
            if gene in geneORGANizer['bodyparts'][organ]:
                stand[gene][organ] = 1 # 先置为0
                cnt_set1 += 1
            else:
                stand[gene][organ] = 0
                cnt_set0 += 1
    print(f"set0_cnt_gene :{cnt_gene}")
    print(f"cnt_set1 :{cnt_set1}")
    print(f"cnt_set0 :{cnt_set0}")

    # 输出每个organ的threshold
    with open('/share/home/liangzhongming/NetOrganizer/RE-ORGANizer-master/generateREGOA/gwas_result/test_base_geneORGANizerGeneOrgang-threshold.txt', 'w') as f:
        f.write('organs\t pre=0.7\t Fmax\t pre=0.3\n')
        cnt_cal = 0
        M = len(geneORGANizer['bodyparts'])
        for tissue in geneORGANizer['bodyparts']:

            threshigh, thresfmax, threslow = testterm([prob[gene][tissue] for gene in genes], [stand[gene][tissue] for gene in genes])
            f.write(tissue+'\t'+str(threshigh)+'\t'+str(thresfmax)+'\t'+str(threslow)+'\n')
            cnt_cal += 1
            print(cnt_cal, '/', M)


def importvec(filename):
    vecs = dict()
    with open('./LiCO/vec/10_15_7path_addGB_4layerTrain_humanGRN_2_vec.txt', 'r') as f:
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
    # 测试是否正确拿到organ名称
    # with open('./datamgi/diffpaths/lzm_test_vecs.txt', 'w') as f:
    #     f.write(str(vecs))
    print('import vecs done')
    return vecs


def main():
    # geneORGANizer = datas.importgeneItem()
    # geneORGANizer = datas.importgeneItem(filename= './process_result/pos.txt')
    # vecs=importvec('./lzm_mp2_result/' + '6_12_0003_7path_17w_sz8_vec.txt')


    # # 参考数据库
    # tf_database_hTFtarget_dict = datas.import_database_hTFtarget()['tfs']
    # tf_database_TRRUST_dict = datas.import_database_TRRUST()['tfs']

    # gene_database_hTFtarget_dict = datas.import_database_hTFtarget()['genes']
    # gene_database_TRRUST_dict = datas.import_database_TRRUST()['tgs']


    vecs = importvec(filename + 'vec/10_15_7path_addGB_4layerTrain_humanGRN_2_vec.txt')
    grn = datas.import_database_grnhg()
    geneORGANizer = datas.importgeneItem()
    grn_tf = datas.import_database_grnhg()['tfs']
    grn_tg = datas.import_database_grnhg()['gene_tfs']
    ppi = datas.importppi()

    res = grn['res'].keys()
    gO_G = datas.import_database_geneORGANizer()['genes']
    gO_B = datas.import_database_geneORGANizer()['bodyparts']


    # 根据GRN中的TF-TG关系，得到TF的阈值
    # lzm_cal1(grn_tf, grn_tg, vecs)
    # print('tf : ', len(grn_tf.keys()))
    # print('tg : ', len(grn_tg.keys()))

    # # 根据geneORGANizer中G-B的关系，得到B的阈值
    # lzm_cal1(res, gO_B, gO_G, vecs)
    lzm_cal_re_g_b(res, gO_B, gO_G, vecs)


if __name__ == '__main__':
    main()