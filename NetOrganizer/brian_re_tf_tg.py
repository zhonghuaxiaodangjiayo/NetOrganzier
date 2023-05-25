import importdatas as datas
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statannotations.Annotator import Annotator
from scipy import stats
import scipy.stats


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

def getthres():

    fmax = dict()

    bodyparts = set()
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

            # print('bodypart : ', obj)
            # print('fmax : ', float(line[index + 1]))

            # fmax[obj] = float(line[2])

            bodyparts.add(obj)  # 得到一系列具有阈值threshold的term

    return fmax, bodyparts


def from_re_search():
    vecs = importvec('./LiCO/' + '10_15_7path_addGB_4layerTrain_humanGRN_2_vec.txt')
    # step1 与小肠 TopSim 的re
    re_set = ['chr1_8997861_9000258', 'chr6_31971978_31973177', 'chr14_104927238_104928863', 'chr19_4296597_4296798', 'chr17_48681409_48682797']

    # step2 找上游TF和下游TG
    grn = datas.importgrnhg()  # return {'genes': genes, 'res': res, 'tfs': tfs}
    # GTEx_database = datas.import_database_GTEx()
    small_intestine_GTEx = datas.import_database_GTEx()
    print('import small_intestine_GTEx done!')

    print('len of re: ', len(small_intestine_GTEx['re_tissue'].keys()))
    print('len of tissue: ', len(small_intestine_GTEx['tissue_re'].keys()))
    print('tissue: ', list(small_intestine_GTEx['tissue_re'].keys()))

    re_tf_dict = dict()
    re_tg_dict = dict()
    for re in re_set:
        if re not in re_tf_dict:
            re_tf_dict[re] = set()
        if re not in re_tg_dict:
            re_tg_dict[re] = set()

        for tf in grn['res'][re]['binded']:
            re_tf_dict[re].add(tf)

        for tg in grn['res'][re]['reg']:
            re_tg_dict[re].add(tg)

    # step3 遍历这个tg,找到它们的re集合,检测这些re和brain的相关性,若相关性较高,考察在GTEx中是否有验证：若没有，因为不可能是RE传给RE，则考察是否是其上游的TF将信息传递过来，而TF则是通过路径TF-TG-B得到信息
    fmax, bodypart = getthres()
    up_stream_re_set = set()
    print('step3: ')
    for re in re_tg_dict:
        for tg in re_tg_dict[re]:
            # 找到该tg的上游re集合，这些re与出发re一样调控了同一个基因，因此检测它们与brain的相关性
            for upStream_re in grn['genes'][tg]:
                sum_of_highSim = 0
                sum_of_lowSim = 0

                up_stream_re_set.add(upStream_re)
                ans = {cosine_similarity([vecs[upStream_re], vecs['small intestine']])[0][1]}
                ans = list(ans)[0]
                # 如何判断其相关性是高低：大于阈值？ 需要引入B的阈值
                if ans > fmax['brain']:
                    # 因为GTEx的RE格式为一个位点，故不能直接比较：需要比对染色体号;再比对位点是否在范围内
                    for gtex_re in small_intestine_GTEx['tissue_re']['brain']:
                        gtex_re = small_intestine_GTEx['tissue_re']['brain'].split('_')
                        gtex_num = gtex_re[0]
                        gtex_pos = gtex_re[1]

                        original_re = upStream_re.split('_')
                        num = original_re[0]
                        start = original_re[1]
                        end = original_re[2]

                        # 在GTEx中被验证
                        if num == gtex_num and gtex_pos >= start and gtex_pos <= end:
                            count_1 = 0
                            len_1 = len(grn['res'][upStream_re]['binded'])
                            # 再找其上游的TF，统计其上游TF与brain的相关性
                            for up_tf in grn['res'][upStream_re]['binded']:
                                ans = {cosine_similarity([vecs[up_tf], vecs['small intestine']])[0][1]}
                                ans = list(ans)[0]

                                # TF与brian相关性高,统计该RE上游TF相关性高的占比，及平均相似度
                                if ans > fmax['small intestine']:
                                    count_1 += 1
                                sum_of_highSim += ans
                            print(str(count_1) + ' / ' + str(len_1) + 'average sim = ' + str(sum_of_highSim/len_1))
                        # 在GTEx中没有被验证
                        else:
                            count_2 = 0
                            len_2 = len(grn['res'][upStream_re]['binded'])
                            # 再找其上游的TF,统计其上游TF与brain的相关性
                            for up_tf in grn['res'][upStream_re]['binded']:
                                ans = {cosine_similarity([vecs[up_tf], vecs['small intestine']])[0][1]}
                                ans = list(ans)[0]

                                sum_of_lowSim += ans
                                # TF与brian相关性高,统计该RE上游TF相关性高的占比，及平均相似度
                                if ans > fmax['small intestine']:
                                    count_2 += 1
                                sum_of_highSim += ans
                            print(str(count_2) + ' / ' + str(len_2) + 'average sim = ' + str(sum_of_lowSim / len_2))
                    # 如果两者相关性存在显著差异，证明给RE注释brian的信息是通过异构网络传递过来的，说明该网络为RE注释B是可信的

    # step4 遍历这个tg,找到它们的re集合,检测这些re和brain的相关性
    # 类似step3的思路，区别在于step3是验证信息从TF传过来;step4是验证信息从TG传过来

# def main():


def my_search():
    vecs = importvec('./LiCO/' + '10_15_7path_addGB_4layerTrain_humanGRN_2_vec.txt')
    fmax, bodypart = getthres()
    # tg_set = []
    tf_set = ['ASCL1', 'BARX2', 'EGR2', 'ESR2', 'IKZF2', 'IRF1', 'IRF4', 'IRF8', 'MEF2B', 'MEF2C']
    re_set = ['chr17_970601_970790', 'chr7_150165290_150165887', 'chr7_134118978_134121205', 'chr11_44123492_44123510']

    grn = datas.importgrnhg()  # return {'genes': genes, 'res': res, 'tfs': tfs}
    geneORGANizer = datas.import_database_geneORGANizer()

    bodypart_predict_re_fmax = datas.import_predict_b_re_fmax()

    tg = 'SLC2A5'
    # for re in grn['genes'][tg]:
    #     ans = {cosine_similarity([vecs[re], vecs['small intestine']])[0][1]}
    #     ans = list(ans)[0]
    #     # print(re)
    #     # 该RE被小肠注释
    #     if ans >= fmax['small intestine']:
    #         print(re + str(ans))

    # for tf in tf_set:
    #     for re in grn['tfs'][tf]:
    #         ans = {cosine_similarity([vecs[re], vecs['small intestine']])[0][1]}
    #         ans = list(ans)[0]
    #
    #         if ans >= fmax['small intestine']:
    #             print(tf + '\t' + re + '\t' + str(ans))

    # for re in re_set:
    #     # 下游tg
    #     # print('re_set : ', len(re_set))
    #     for tg in grn['res'][re]['reg']:
    #         # 下游tg找其上游re
    #         # print('tgs len : ', len(grn['res'][re]['reg']))
    #         for upStreamRe in grn['genes'][tg]:
    #             # print('res len : ', len(grn['genes'][tg]))
    #             # 判定该RE与指定tissue的相关性
    #             ans = {cosine_similarity([vecs[upStreamRe], vecs['brain']])[0][1]}
    #             ans = list(ans)[0]
    #             # print(re + '\t' + tg + '\t' + upStreamRe + '\t' + str(ans))
    #
    #             if ans >= fmax['brain']:
    #                 print(re + '\t' + tg + '\t' + upStreamRe + '\t' + str(ans))

    with open('./LiCO/var/Box_ReTfSim.txt', 'w') as f:

        allRelated_RE_TF = []
        allUnrelated_RE_TF = []
        all_rela_cnt = 0
        all_unrela_cnt = 0
        cnt_bodypart = 0
        for bodypart in bodypart_predict_re_fmax['bodyparts']:
            cnt_bodypart += 1
            print('bodypart : ', cnt_bodypart)
            for re in bodypart_predict_re_fmax['bodyparts'][bodypart]:
                # 统计其上游TF与当前bodypart的注释情况/相关性——>比对箱线图
                relatedRE_relatedTFs_dict = []
                relatedRE_totalTFs_dict = []

                for upStreamTF_1 in grn['res'][re]['binded']:
                    ans = {cosine_similarity([vecs[upStreamTF_1], vecs[bodypart]])[0][1]}
                    ans = list(ans)[0]
                    relatedRE_totalTFs_dict.append(ans)
                    allRelated_RE_TF.append(ans)
                    all_rela_cnt += 1
                    if all_rela_cnt % 100 == 0:
                        print('all_rela_cnt=', all_rela_cnt)
                    if ans > fmax[bodypart]:
                        relatedRE_relatedTFs_dict.append(ans)
                # 找到调控的G
                for gene in grn['res'][re]['reg']:
                    # 调控同一G且为注释B的RE，其TFs的数据超过当前RE的数量
                    unrelatedRE_totalTFs_dict = []
                    unrelatedRE_relatedTFs_dict = []
                    # 找到上游的RE
                    for upStreamRe in grn['genes'][gene]:
                        if upStreamRe != re:
                            # 衡量是否被当前bodypart注释
                            ans = {cosine_similarity([vecs[upStreamRe], vecs[bodypart]])[0][1]}
                            ans = list(ans)[0]
                            # 若没有被注释，找其上游的TF
                            if ans < fmax[bodypart]:
                                # f.write('unrelated RE : ' + upStreamRe + '\n')
                                for upStreamTF_2 in grn['res'][upStreamRe]['binded']:
                                    # 统计其上游TF与bodypart的注释情况/相关性
                                    ans = {cosine_similarity([vecs[upStreamTF_2], vecs[bodypart]])[0][1]}
                                    ans = list(ans)[0]
                                    unrelatedRE_totalTFs_dict.append(ans)
                                    allUnrelated_RE_TF.append(ans)
                                    all_unrela_cnt += 1
                                    if all_unrela_cnt % 10000 == 0:
                                        print('all_unrela_cnt=', all_unrela_cnt)

                                    if ans > fmax[bodypart]:
                                        unrelatedRE_relatedTFs_dict.append(ans)
                    # if relatedRE_totalTFs_dict and unrelatedRE_totalTFs_dict:
                    #     # 绘制箱线图
                    #     figName = str(bodypart) + '_' + str(re) + '_' + str(gene)
                    #     savePath = './LiCO/fig/B_RE_TF/' + figName
                    #     calBox(relatedRE_totalTFs_dict, unrelatedRE_totalTFs_dict, savePath)

        figName2 = 'all_relatedRETF_UnrelatedRETF'
        savePath2 = './LiCO/fig/B_RE_TF/total/' + figName2
        calBox(allRelated_RE_TF, allUnrelated_RE_TF, savePath2)


def test():
    bodypart_predict_re_fmax = datas.import_predict_b_re_fmax()
    grn = datas.importgrnhg()  # return {'genes': genes, 'res': res, 'tfs': tfs}

    # cnt_bodypart = 0
    #
    # for bodypart in bodypart_predict_re_fmax['bodyparts']:
    #     cnt_bodypart += 1
    #     print('bodypart_cnt : ', cnt_bodypart)
    #     print('bodypart:', bodypart)
    #     for re in bodypart_predict_re_fmax['bodyparts'][bodypart]:
    #         print('re:', re)
    #         for upStreamTF_1 in grn['res'][re]['binded']:
    #             print('tf:', upStreamTF_1)

    print(bodypart_predict_re_fmax['bodyparts']['vocal cord'])

def calBox(relatedRE_totalTFs_dict, unrelatedRE_totalTFs_dict, savePath):


    # 做标准化
    # z-score
    zscore_scaler = preprocessing.StandardScaler()
    # d1 = np.array(relatedRE_relatedTFs_dict).reshape(-1, 1)  # 1规定列数为1，-1表根据给定的列数自动分配行数
    d2 = np.array(relatedRE_totalTFs_dict, dtype=object).reshape(-1, 1)
    # d3 = np.array(unrelatedRE_relatedTFs_dict).reshape(-1, 1)
    d4 = np.array(unrelatedRE_totalTFs_dict, dtype=object).reshape(-1, 1)

    # t_test
    t, pval = t_test(d2, d4)
    print("raw:")
    print(t, pval)

    # s1 = zscore_scaler.fit_transform(d1)
    s2 = zscore_scaler.fit_transform(d2)
    # s3 = zscore_scaler.fit_transform(d3)
    s4 = zscore_scaler.fit_transform(d4)

    t, pval = t_test(s2, s4)
    print("zscore:")
    print(t, pval)

    # print('original: ', not_in_grn_TFtarget)
    # print('transformed: ', data_score_1)

    # # max-min
    minmax_scaler = preprocessing.MinMaxScaler()
    # min1 = minmax_scaler.fit_transform(d1)
    min2 = minmax_scaler.fit_transform(d2)
    # min3 = minmax_scaler.fit_transform(d3)
    min4 = minmax_scaler.fit_transform(d4)

    t, pval = t_test(min2, min4)
    print("max_min:")
    print(t, pval)

    # maxAbs
    maxabs_scaler = preprocessing.MaxAbsScaler()
    # max1 = maxabs_scaler.fit_transform(d1)
    max2 = maxabs_scaler.fit_transform(d2)
    # max3 = maxabs_scaler.fit_transform(d3)
    max4 = maxabs_scaler.fit_transform(d4)

    t, pval = t_test(max2, max4)
    print("maxAbs:")
    print(t, pval)

    # robustScaler
    Robust_scaler = preprocessing.RobustScaler()
    # r1 = Robust_scaler.fit_transform(d1)
    r2 = Robust_scaler.fit_transform(d2)
    # r3 = Robust_scaler.fit_transform(d3)
    r4 = Robust_scaler.fit_transform(d4)

    t, pval = t_test(r2, r4)
    print("robustScaler:")
    print(t, pval)

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
    plt.yticks(np.arange(-2, 2.2, 0.2))
    plt.savefig(savePath + ".png", dpi=300)
    # plt.show()
    plt.close()




def t_test(e, c):
    t, pval = scipy.stats.ttest_ind(e, c)
    return t, pval



if __name__ == '__main__':
    # main()
    # from_re_search()
    my_search()
    # test()
    # calBox_pvalue()