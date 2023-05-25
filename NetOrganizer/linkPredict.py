import importdatas as datas
from sklearn.metrics.pairwise import cosine_similarity
from multiprocessing import Process, Queue, Manager
import random
import pickle


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


def lzm_calbodyparts(redict, vecs, bodyparts, geneORGANizer, logo):
    M = len(bodyparts)
    count = 0
    prob = dict()
    # 在调用之后再写
    # gfmax = open('./datamgi/' + 'organs_' + 're-organ-fmax.txt', 'w')

    for bodypartA in bodyparts:
        prob[bodypartA] = dict()
        for bodypartB in geneORGANizer['bodyparts'].keys():
            print(bodypartA)
            print(bodypartB)
            prob[bodypartA][bodypartB] = cosine_similarity([vecs[bodypartA], vecs[bodypartB]])[0][1]
        # gfmax.write(organ + '\t')

        annos = set()
        for bodypartB in geneORGANizer['bodyparts'].keys():
            # if prob[bodypartA][bodypartB] > 0.292:
            #     # gfmax.write(organ + '\t')
            #     annos.add(bodypartB)
            if prob[bodypartA][bodypartB] >= 0.8:
                # gfmax.write(organ + '\t')
                annos.add(bodypartB)
        # gfmax.write('\n')
        count += 1
        redict[bodypartA] = annos

        if count % 20 == 0:
            print(logo, count, '/', M, ' lzm_calbodyparts()')


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


def main():
    '''
    global grn
    grn = datas.importgrn()  # return {'genes':{gene:re,gene:re...},'res':{re:{'reg':genes,'bind':tfs}},'tfs':{tf:re,tf:re}}
    print('import grn done')
    '''
    global geneORGANizer
    # geneORGANizer = datas.importgene_organs('./New_gene_organs_2.txt')
    geneORGANizer = datas.importgeneItem()

    # vecs = importvec(path+'diffpaths/' + 'genes_organs(5metapath)_' + 'vecs'+notename+'.txt')
    vecs = importvec('./model0.87/classical/' + 'classical_add_GL53w_GR70w_GS87w_GO140w_vec.txt')
    # res = grn['res'].keys()
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

if __name__=='__main__':
    main()