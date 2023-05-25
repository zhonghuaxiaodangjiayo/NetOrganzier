import importdatas as datas
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np

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

# 取得B的阈值
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

# 导入与geneORGANizer/hGraph重合的高原适应症受选择G
def import_altitude_genes(opt):
    geneORGANizer = datas.import_database_geneORGANizer()
    grn = datas.import_database_grnhg()
    ppi = datas.importppi()
    total_geneset = (set(geneORGANizer['genes'].keys()).union(set(ppi.keys()))).union(set(grn['gene_tfs'].keys()))

    Genes_under_selection = []
    # if opt == 'all':
    cnt_all = 0
    with open('./LiCO/data/Indications_for_high_altitude_genes_all.txt', 'r') as f:
        print('open _all txt')
        for line in f:
            line = line.split('\n')
            gene = line[0].upper()
            # if gene in geneORGANizer['genes']:
            if gene in geneORGANizer['genes']:
                Genes_under_selection.append(gene)
                cnt_all += 1
    print('cnt_all:', cnt_all)
    # if opt == 'more':
    #     cnt_more = 0
    #     with open('./LiCO/data/Indications_for_high_altitude_genes_more.txt', 'r') as f:  # 不包含只有1次文献验证的G
    #         print('open _more txt')
    #         for line in f:
    #             line = line.split('\n')
    #             gene = line[0].upper()
    #             if gene in geneORGANizer['genes']:
    #                 Genes_under_selection.append(gene)
    #                 cnt_more += 1
    #     print('cnt_more:', cnt_more)

    return Genes_under_selection

# 导入与geneORGANizer/hGraph重合的糖尿病有关的G
def import_T2D_diabetes_gene():
    cnt = 0
    # filename = './LiCO/diabetes_mellitus/T2D/associations.csv'
    # with open(filename, 'r') as f:
    #     next(f)
    #     for line in f:
    #
    #         line = line.split(',')
    #         print(len(line))
    #         #
    #         # raw_re = line[0] # "10:114754088:T:C"
    #         # raw_re = eval(raw_re)
    #         # raw_re = raw_re.split(':')
    #         # chr = raw_re[0]
    #         #
    #         # pos = raw_re[1]
    #         #
    #         #
    #         # p_val = line[10]
    #         #
    #         #
    #         # nearest_gene = line[26]
    #         # nearest_gene = nearest_gene.replace('[','').replace(']','')
    #         # nearest_gene = eval(nearest_gene)
    #         #
    #         # if float(p_val) <= 0.00000005:
    #         #     cnt += 1
    #     print(cnt)
    # f.close()

    # # 写入受选择gene
    # cnt = 0
    # filename = './LiCO/diabetes_mellitus/T2D/gene_table.csv'
    # w = open('./LiCO/diabetes_mellitus/T2D/gene_5e9.txt', 'w')
    # with open(filename, 'r') as f:
    #     next(f)
    #     for line in f:
    #
    #         line = line.split(',')
    #         # print(len(line))
    #
    #         gene = eval(line[1])
    #
    #
    #         p_val = line[6]
    #
    #         if float(p_val) <= 0.000000005:
    #             w.write(gene)
    #             w.write('\n')
    #             cnt += 1
    #     print(cnt)
    # f.close()
    # w.close()

    # 返回受选择基因
    geneORGANizer = datas.import_database_geneORGANizer()
    grn = datas.import_database_grnhg()
    ppi = datas.importppi()
    total_geneset = (set(geneORGANizer['genes'].keys()).union(set(ppi.keys()))).union(set(grn['gene_tfs'].keys()))

    Genes_under_selection = []
    cnt = 0
    with open('./LiCO/diabetes_mellitus/T2D/gene_5e8.txt', 'r') as f:
        for line in f:
            line = line.split()
            gene = line[0]
            if gene in geneORGANizer['genes']:
            # if gene in total_geneset:
                if gene not in Genes_under_selection:
                    Genes_under_selection.append(gene)
                    cnt += 1
    print('cnt_T2D : ', cnt)
    f.close()

    return Genes_under_selection

# 导入与geneORGANizer/hGraph重合的回声定位受选择G
def import_echo_genes():
    geneORGANizer = datas.import_database_geneORGANizer()
    grn = datas.import_database_grnhg()
    ppi = datas.importppi()
    total_geneset = (set(geneORGANizer['genes'].keys()).union(set(ppi.keys()))).union(set(grn['gene_tfs'].keys()))

    Genes_under_selection = []
    cnt = 0
    with open('./LiCO/echo/gene.txt', 'r') as f:
        for line in f:
            line = line.split('\n')
            gene = line[0].upper()
            # print(gene)
            # if gene in total_geneset:
            if gene in geneORGANizer['genes']:
                Genes_under_selection.append(gene)
                cnt += 1
    print('cnt_echo_gene: ', cnt)

    return Genes_under_selection


def find_organs_background_genes():
    geneORGANizer = datas.import_database_geneORGANizer()  # return {'genes': genes, 'bodyparts': bodyparts}
    vecs = importvec('./LiCO/' + '10_15_7path_addGB_4layerTrain_humanGRN_2_vec.txt')
    Genes_under_selection = import_altitude_genes(opt='all')  # 所有文献记载过的G


    # 找placenta、thymus、meninges
    placenta_geneSets = set()
    placenta_UnrelatedGeneSets = set()

    thymus_geneSets = set()
    thymus_UnrelatedGeneSets = set()

    meninges_geneSets = set()
    meninges_UnrelatedGeneSets = set()



    # 将63个交集G分为两部分：有直接联系、无直接联系
    for gene in Genes_under_selection:
        if gene in geneORGANizer['bodyparts']['placenta']:
            placenta_geneSets.add(gene)
        else:
            placenta_UnrelatedGeneSets.add(gene)

        if gene in geneORGANizer['bodyparts']['thymus']:
            thymus_geneSets.add(gene)
        else:
            thymus_UnrelatedGeneSets.add(gene)

        if gene in geneORGANizer['bodyparts']['meninges']:
            meninges_geneSets.add(gene)
        else:
            meninges_UnrelatedGeneSets.add(gene)

    # 分别计算相似度，然后通过箱线图比较
    related_gene = []
    unrelated_gene = dict()
    for gene in placenta_geneSets:
        ans = {cosine_similarity([vecs[gene], vecs['placenta']])[0][1]}
        ans = list(ans)[0]

        related_gene.append(ans)

    # 取出TopN重点研究
    for gene in placenta_UnrelatedGeneSets:
        ans = {cosine_similarity([vecs[gene], vecs['placenta']])[0][1]}
        ans = list(ans)[0]

        if gene not in unrelated_gene:
            unrelated_gene[gene] = ans


    unrelated_gene_order = sorted(unrelated_gene.items(), key=lambda x: x[1], reverse=True)
    cnt = 0
    for item in unrelated_gene_order:
        cnt += 1
        print(item[0])
        if cnt == 10:
            break


    # # 绘制箱线图
    # labels = 'related_gene', 'unrelated_gene'
    #
    # plt.grid(True)  # 显示网格
    # plt.boxplot(
    #     [related_gene, unrelated_gene],
    #     medianprops={'color': 'red', 'linewidth': '1.2'},
    #     meanline=True,
    #     showmeans=True,
    #     meanprops={'color': 'blue', 'ls': '--', 'linewidth': '1.2'},
    #     flierprops={"marker": "o", "markerfacecolor": "red", "markersize": 3},
    #     labels=labels)
    # plt.yticks(np.arange(0, 1.0, 0.1))
    # plt.savefig('./LiCO/organs_bg_gene/' + "placenta.png", dpi=500)
    # # plt.show()
    # plt.close()

def find_gene_up_downStream():
    # Genes_under_selection = import_altitude_genes(opt='all')  # 所有文献记载过的G
    # Genes_under_selection = import_T2D_diabetes_gene()
    Genes_under_selection = import_echo_genes()
    geneORGANizer = datas.import_database_geneORGANizer()  # return {'genes': genes, 'bodyparts': bodyparts}
    grn = datas.import_database_grnhg()

    vecs = importvec('./LiCO/' + '10_15_7path_addGB_4layerTrain_humanGRN_2_vec.txt')
    upStream = []
    downStream = set()
    fmax, bodyparts = getthres()

    w1 = open('./LiCO/organs_bg_gene/echo/larynx_A1.txt', 'w')
    w2 = open('./LiCO/organs_bg_gene/echo/larynx_A2.txt', 'w')
    w3 = open('./LiCO/organs_bg_gene/echo/larynx_B1.txt', 'w')
    w4 = open('./LiCO/organs_bg_gene/echo/larynx_B2.txt', 'w')

    #################################################################################################
    # 直接联系 VS 直接联系 + 背后TF、RE
    # placenta_geneSets = set()
    # Unthymus_geneSets = set()
    # Unmeninges_geneSets = set()
    pancreas_geneSets = set()
    Unpancreas_geneSets = set()
    cnt_related = 0
    cnt_Unrelated = 0
    for gene in Genes_under_selection:
        # 63个G不一定都在grn中
        if gene in grn['gene_tfs']:
        # if gene in grn['tfs']:
        # if True:
            # cnt += 1
            if gene in geneORGANizer['bodyparts']['larynx']:
                pancreas_geneSets.add(gene)
                cnt_related += 1

            if gene not in geneORGANizer['bodyparts']['larynx']:
                Unpancreas_geneSets.add(gene)
                cnt_Unrelated += 1
    print('30 in grn and relate to organ: ', cnt_related)
    print('30 in grn and Unrelate to organ: ', cnt_Unrelated)

    # 非直接联系 VS 非直接联系 + 背后调控机制
    # Notplacenta_geneSets = set()
    # cnt = 0
    # for gene in Genes_under_selection:
    #     # 63个G不一定都在grn中
    #     if gene in grn['gene_tfs']:
    #     # if gene in grn['tfs']:
    #     # if True:
    #         # cnt += 1
    #         if gene not in geneORGANizer['bodyparts']['placenta']:
    #             Notplacenta_geneSets.add(gene)
    #             cnt += 1
    # print('63 in grn : ', cnt)

    related_gene_sim = []
    related_genePlus_sim = []


    # 分别计算相似度，然后通过箱线图比较
    for gene in pancreas_geneSets:
        ans = {cosine_similarity([vecs[gene], vecs['larynx']])[0][1]}
        ans = list(ans)[0]

        related_gene_sim.append(ans)
        w1.write(str(ans))
        w1.write('\n')
        # related_genePlus_sim.append(ans)

    for gene in Unpancreas_geneSets:
        ans = {cosine_similarity([vecs[gene], vecs['larynx']])[0][1]}
        ans = list(ans)[0]

        related_gene_sim.append(ans)
        w3.write(str(ans))
        w3.write('\n')


    # 由基因找上游TF、RE
    cnt_tf = 0
    cnt_re = 0
    for gene in pancreas_geneSets:

        for tf in grn['gene_tfs'][gene]:

            ans = {cosine_similarity([vecs[tf], vecs['larynx']])[0][1]}
            ans = list(ans)[0]

            if ans >= fmax['larynx']:
                related_genePlus_sim.append(ans)
                cnt_tf += 1
                w2.write(str(ans))
                w2.write('\n')

        for re in grn['gene_res'][gene]:
            # print('len of gene2re : ', len(grn['gene_res'][gene]))


            ans = {cosine_similarity([vecs[re], vecs['larynx']])[0][1]}
            ans = list(ans)[0]

            if ans >= fmax['larynx']:
                related_genePlus_sim.append(ans)
                cnt_re += 1
                w2.write(str(ans))
                w2.write('\n')

    print('cnt_tf : ', cnt_tf)
    print('cnt_re : ', cnt_re)

    w1.close()
    w2.close()


    # 由基因找上游TF、RE
    cnt_tf = 0
    cnt_re = 0
    for gene in Unpancreas_geneSets:

        for tf in grn['gene_tfs'][gene]:

            ans = {cosine_similarity([vecs[tf], vecs['larynx']])[0][1]}
            ans = list(ans)[0]

            if ans >= fmax['larynx']:
                related_genePlus_sim.append(ans)
                cnt_tf += 1
                w4.write(str(ans))
                w4.write('\n')

        for re in grn['gene_res'][gene]:
            # print('len of gene2re : ', len(grn['gene_res'][gene]))

            ans = {cosine_similarity([vecs[re], vecs['larynx']])[0][1]}
            ans = list(ans)[0]

            if ans >= fmax['larynx']:
                related_genePlus_sim.append(ans)
                cnt_re += 1
                w4.write(str(ans))
                w4.write('\n')
    print('cnt_tf : ', cnt_tf)
    print('cnt_re : ', cnt_re)

    w3.close()
    w4.close()


    # # 由基因找下游TG、RE（把gene当TF）
    # cnt_tg = 0
    # cnt_re = 0
    # for tf in Notplacenta_geneSets:
    #
    #     if tf in grn['tfs']:
    #         for tg in grn['tfs'][tf]:
    #             if tg in vecs:
    #                 cnt_tg += 1
    #
    #                 ans = {cosine_similarity([vecs[tg], vecs['placenta']])[0][1]}
    #                 ans = list(ans)[0]
    #
    #                 related_genePlus_sim.append(ans)
    #
    #
    #     if tf in grn['tfs']:
    #         for re in grn['tf_res'][tf]:
    #             cnt_re += 1
    #
    #             ans = {cosine_similarity([vecs[re], vecs['placenta']])[0][1]}
    #             ans = list(ans)[0]
    #
    #             related_genePlus_sim.append(ans)



    # # 绘制箱线图
    # labels = 'UnRelated_gene_sim', 'UpStream_fmaxSim'
    #
    # plt.grid(True)  # 显示网格
    # plt.boxplot(
    #     [related_gene_sim, related_genePlus_sim],
    #     medianprops={'color': 'red', 'linewidth': '1.2'},
    #     meanline=True,
    #     showmeans=True,
    #     meanprops={'color': 'blue', 'ls': '--', 'linewidth': '1.2'},
    #     flierprops={"marker": "o", "markerfacecolor": "red", "markersize": 3},
    #     labels=labels)
    # plt.yticks(np.arange(0, 1.0, 0.1))
    # plt.savefig('./LiCO/organs_bg_gene/' + "placenta_Unrelated_UpStream_fmaxSim.png", dpi=500)
    # # plt.show()
    # plt.close()

def find_key_grn():
    # Genes_under_selection = import_altitude_genes(opt='all')  # 所有文献记载过的G
    # Genes_under_selection =
    geneORGANizer = datas.import_database_geneORGANizer()  # return {'genes': genes, 'bodyparts': bodyparts}
    grn = datas.import_database_grnhg()

    vecs = importvec('./LiCO/' + '10_15_7path_addGB_4layerTrain_humanGRN_2_vec.txt')
    fmax, bodyparts = getthres()

    Unmeninges_geneSets = set()
    meninges_geneSets = set()
    cnt_in = 0
    cnt_not = 0
    for gene in Genes_under_selection:
        # 63个G不一定都在grn中
        if gene in grn['gene_tfs']:
        # if gene in grn['tfs']:
        # if True:
            # cnt += 1
            if gene in geneORGANizer['bodyparts']['meninges']:
                if gene == 'EPAS1':
                    print('EPAS1 in')
                meninges_geneSets.add(gene)
                cnt_in += 1
            elif gene not in geneORGANizer['bodyparts']['meninges']:
                if gene == 'EPAS1':
                    print('EPAS1 not in ')
                Unmeninges_geneSets.add(gene)
                cnt_not += 1
    print('63 in grn and relate to organ: ', cnt_in)
    print('63 in grn and Unrelate to organ: ', cnt_not)


    cnt_tf = 0
    cnt_re = 0
    related_tf = dict()
    re_dict = dict()
    w2 = open('./LiCO/organs_bg_gene/Unplacenta_related_gene_tf_re.txt', 'w')

    for gene in meninges_geneSets:
        # if gene not in related_tf:
        #     related_tf[gene] = dict()
        # if gene not in related_re:
        #     related_re[gene] = dict()

        for tf in grn['gene_tfs'][gene]:

            ans = {cosine_similarity([vecs[tf], vecs['meninges']])[0][1]}
            ans = list(ans)[0]

            if ans >= fmax['meninges']:
                # if gene not in related_tf:
                #     related_tf[gene] = set()
                # related_tf[gene].add(tf)
                if tf not in related_tf:
                    related_tf[tf] = set()

                cnt_tf += 1
                # print('current gene : ', gene)
                # print('current tf : ', tf)
    print(cnt_tf)
    print(related_tf)

    #     # 试着由G找RE
    #     for re in grn['gene_res'][gene]:
    #         ans = {cosine_similarity([vecs[re], vecs['meninges']])[0][1]}
    #         ans = list(ans)[0]
    #
    #         if ans >= fmax['meninges']:
    #             # if gene not in related_tf:
    #             #     related_tf[gene] = set()
    #             # related_tf[gene].add(tf)
    #             if re not in re_dict:
    #                 related_tf[re] = set()
    #
    #             cnt_re += 1
    #             # print('current gene : ', gene)
    #             # print('current tf : ', tf)
    # print('cnt_re : ', cnt_re)

    related_re = dict()
    cnt_re = 0
    # 由TF找RE
    print('len of related_tf: ', len(related_tf))
    for tf in related_tf:
        for re in grn['tf_res'][tf]:
            ans = {cosine_similarity([vecs[re], vecs['meninges']])[0][1]}
            ans = list(ans)[0]

            if ans >= fmax['meninges']:
                if tf not in related_re:
                    related_re[tf] = set()
                related_re[tf].add(re)
                cnt_re += 1
                # print(ans)
                for gene in meninges_geneSets:
                    # print(grn['gene_res'][gene])
                    if re in grn['gene_res'][gene]:
                        print('cur gene:', gene)
                        print('cur re:', re)

    print('cnt_re : ', cnt_re)
    # print(related_re)



        # for re in grn['gene_res'][gene]:
        #
        #     ans = {cosine_similarity([vecs[re], vecs['meninges']])[0][1]}
        #     ans = list(ans)[0]
        #
        #     if ans >= fmax['meninges'] - 0.1:
        #         related_re[gene][re] = ans
        #
        #     # related_re[gene][re] = ans


    # related_tf_sort = sorted(related_tf.items(), key=lambda x: x[1], reverse=True)
    # related_re_sort = sorted(related_re.items(), key=lambda x: x[1], reverse=True)

    # cnt = 0
    # print('tf:')
    # for key in related_tf:
    #     print(str(key))
    #     for value in related_tf[key]:
    #         print(value)
    #     # cnt += 1
    #     # if cnt == 10:
    #     #     break
    # cnt = 0
    # print('re:')
    # for key in related_re:
    #     print(str(key))
    #     for value in related_re[key]:
    #         print(value)
    #     # cnt += 1
    #     # if cnt == 10:
    #     #     break


if __name__ == '__main__':
    # find_organs_background_genes()
    find_gene_up_downStream()
    # find_key_grn()