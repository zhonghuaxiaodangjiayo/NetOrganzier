import importdatas as datas
from sklearn.metrics.pairwise import cosine_similarity
Function = 'CC'
from multiprocessing import Process, Queue, Manager
import random
import pickle

path = './datamgi/'
notename = '-noDis'

# def importvec(filename):
#     vecs=dict()
#     with open(filename, 'r') as f:
#         next(f)
#         for line in f:
#             line = line.split()
#             vecs[line[0]] = [float(x) for x in line[1:]]
#     print('import vecs done')
#     return vecs

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

# 由threshold.txt文件得到fmax，即[term————threshold]
def getthres():
    fmax = dict()
    bodyparts = set()
    tfs = set()
    with open('./LiCO/threshold/' + 'gO_B_G_threshold.txt', 'r') as f:  # 由文件'calthres.py'中的方法cal()得到，其中是term和对应的threshold
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

# 最终生成re-organ文件
def calres(vecs, res, fmax, goaterms):  # vecs存向量  res存RE  fmax存term的thres goaterms哪来的?
    M = len(res)
    count = 0
    prob = dict()

    gfmax = open('./datamgi/diffpaths/6metapath_add_OGOG/' + 'organs_' + 're-organ-fmax.txt', 'w')  # 写成"organ——goTerm goTerm ..."形式

    for re in res:
        prob[re] = dict()
        for term in go.keys():
            prob[re][term] = cosine_similarity([vecs[term], vecs[re]])[0][1]  # prob中存re和term之间余弦相似度
        terms = set(go.keys())
        while terms:  # 取出go中的term
            term = terms.pop()
            nowmax = prob[re][term]
            nowterm = term
            #  找当前、子孙中的最大
            for tterm in go[term]['descent'] & set(prob[re].keys()):
                if prob[re][tterm] > nowmax:
                    nowmax = prob[re][tterm]
                    nowterm = tterm
            prob[re][term] = nowmax

            for tterm in go[term]['descent'] & go[nowterm]['ancesent'] & terms:
                prob[re][tterm] = nowmax
            terms = terms - go[term]['descent'] & go[nowterm]['ancesent']
        # 写re
        gfmax.write(re+'\t')
        # 写大于阈值的 term
        for term in goaterms:
            if prob[re][term] > fmax[term]:
                gfmax.write(term + '\t')

        gfmax.write('\n')

        count += 1
        print(count, '/', M, ' test')


def lzm_calres(vecs, bodyparts, fmax, re_set):
    M = len(bodyparts)
    print('len of bodyparts : ', len(bodyparts))
    count = 0
    prob = dict()

    # gfmax = open('./process_result/' + 'bodyparts_' + 're-bodypart-fmax.txt', 'w')
    with open('./gwas_result/' + 'organ_RE_fmax-full10.txt', 'w') as w:
        for bodypart in bodyparts:
            re_found = 0
            re_list = []
            prob[bodypart] = dict()
            for re in re_set:
                if re in vecs:
                    prob[bodypart][re] = cosine_similarity([vecs[re], vecs[bodypart]])[0][1]
                    if prob[bodypart][re] >= fmax[bodypart]:
                        re_found += 1
                        re_list.append((re,prob[bodypart][re]))
            
            if re_found >= 10:
                re_list.sort(reverse=True, key=lambda x: x[1])
                re_list = [x[0] for x in re_list]

            else:
                re_list.sort(reverse=True, key=lambda x: x[1])
                re_list = [x[0] for x in re_list]
                while len(re_list) < 10:
                    missing_count = 10 - len(re_list)
                    all_re = [(re,prob[bodypart][re]) for re in prob[bodypart] if prob[bodypart][re] > 0]
                    all_re.sort(reverse=True, key=lambda x: x[1])
                    for r in all_re:
                        if r[0] not in re_list:
                            re_list.append(r[0])
                            missing_count -= 1
                            if missing_count == 0:
                                break


            w.write(bodypart + '\t')
            # print('had wrote tf : ', tf)
            for re in re_list:
                w.write(re + '\t')
            w.write('\n')

            count += 1
            print(count, '/', M, ' calres()')


# 区别于calres()：用多进程方式运行
def calress(redict, vecs, res, fmax, goaterms, logo):
    M = len(res)
    count = 0
    prob = dict()
    # g = open('./data' + logo+Function + 're-term-low.txt', 'w')
    for re in res:  # 一系列具有threshold的re
        prob[re] = dict()
        # g.write(re+'\t')
        for term in go.keys() & vecs.keys():  # 取出在go中存在且具有向量表示的每一个term
            prob[re][term] = cosine_similarity([vecs[term], vecs[re]])[0][1]  # 满足条件的term都与当前re计算余弦相似度
        terms = set(go.keys() & vecs.keys())  # 并把term存入集合set
        while terms:
            term = terms.pop()
            nowmax = prob[re][term]
            nowterm = term
            # 更新：目前term的子孙也许与re有更大的余弦相似度
            for tterm in go[term]['descent'] & set(prob[re].keys()):
                if prob[re][tterm] > nowmax:
                    nowmax = prob[re][tterm]
                    nowterm = tterm
            prob[re][term] = nowmax
            # 更新：
            for tterm in go[term]['descent'] & go[nowterm]['ancesent'] & terms:
                prob[re][tterm] = nowmax

            #
            terms = terms - go[term]['descent'] & go[nowterm]['ancesent']

        # 取交集操作 大于阈值的term才能纳入
        # 区别于calres()的部分，
        annos = set()
        for term in goaterms:
            if prob[re][term] > fmax[term]:
                annos.add(term)
                #print(term)
                #g.write(term+'\t')
        count += 1
        #g.write('\n')
        redict[re] = annos
        #for term in redict[re]:
            #print(term)
        #print(re + '\t' + '\t'.join(list(redict[re]))+'\n')
        if count % 20 == 0:
            print(logo, count, '/', M, ' test')



def lzm_calress(redict, vecs, tfs, fmax, grn, logo):
    M = len(tfs)
    count = 0
    prob = dict()
    # 在调用之后再写
    # gfmax = open('./datamgi/' + 'organs_' + 're-organ-fmax.txt', 'w')

    for tf in tfs:
        prob[tf] = dict()
        for tg in grn['gene_tfs'].keys():
            if tg in vecs:
                prob[tf][tg] = cosine_similarity([vecs[tg], vecs[tf]])[0][1]
        # gfmax.write(organ + '\t')

        annos = set()
        for tg in grn['gene_tfs'].keys():
            if tg in vecs:
                if prob[tf][tg] > fmax[tf]:
                    # gfmax.write(organ + '\t')
                    annos.add(tg)
                    # print('now tf : ' + str(tf) + ', add tg : ' + str(tg))
        # gfmax.write('\n')
        count += 1
        redict[tf] = annos

        if count % 20 == 0:
            print(logo, count, '/', M, ' lzm_calress()')



def main():
    '''
    global grn
    grn = datas.importgrn()  # return {'genes':{gene:re,gene:re...},'res':{re:{'reg':genes,'bind':tfs}},'tfs':{tf:re,tf:re}}
    print('import grn done')
    '''
    # global geneORGANizer
    # geneORGANizer = datas.importgene_organs('./New_gene_organs_2.txt')
    # geneORGANizer = datas.importgeneItem()

    geneORGANizer = datas.importgeneItem()
    # grn = datas.import_database_grnhg()
    # grn = datas.importgrnhg()
    grn = datas.import_database_grnhg()
    ppi = datas.importppi()

    # vecs = importvec(path+'diffpaths/' + 'genes_organs(5metapath)_' + 'vecs'+notename+'.txt')
    # vecs = importvec('./process_result/' + 'full7pathModel_vec.txt')
    vecs = importvec('./LiCO/' + '10_15_7path_addGB_4layerTrain_humanGRN_2_vec.txt')

    # res = grn['res'].keys()

    # g = open(path + 'REsP3.pkl', 'rb+')
    # res = pickle.load(g)
    # res = res[6000:7000]

    # 读取tf的阈值
    # fmax, tfs = getthres()
    # 读取bodypart的阈值
    fmax, bodyparts = getthres()

    # calres(vecs, res, high, fmax, low, terms)


    # total_geneset = (set(geneORGANizer['genes'].keys()).union(set(ppi.keys()))).union(set(grn['gene_tfs'].keys()))
    re_set = grn['res'].keys()
    # 不采用多进程的方式
    lzm_calres(vecs, bodyparts, fmax, re_set)

    # redict = Manager().dict()
    # N = 8
    # resset = []
    # for i in range(N):
    #     resset.append(set())
    # for i in tfs:
    #     resset[random.randint(0, N - 1)].add(i)
    # processes = []
    # for i in range(N):
    #     processes.append(Process(target=lzm_calress, args=(redict, vecs, resset[i], fmax, grn, 'p' + str(i))))
    # for i in range(N):
    #     processes[i].start()
    # for i in range(N):
    #     processes[i].join()
    # with open('./LiCO/threshold/' + 'grn_tf_tg_fmax.txt', 'a') as f:  # 'a' 文件不存在也会创建写入
    #     for tf in redict.keys():
    #         f.write(tf + '\t' + '\t'.join(list(redict[tf]))+'\n')

if __name__=='__main__':
    main()
    # getthres()