from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import openpyxl

from sklearn.metrics.pairwise import cosine_similarity

# import sys 
# sys.path.append("..") 
import importdatas as datas

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
    # re-g-b-threshold.txt
    with open('/share/home/liangzhongming/RE_ORGANize_4layers/RE-ORGANizer-master/generateREGOA/LiCO/threshold/' + 'gO_B_G_threshold.txt', 'r') as f:  # 由文件'calthres.py'中的方法cal()得到，其中是term和对应的threshold
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

def handle():
    filename = '/share/home/liangzhongming/NetOrganizer/RE-ORGANizer-master/generateREGOA/NO_plusAlc/echo/gene.txt'
    filename20 = '/share/home/liangzhongming/NetOrganizer/RE-ORGANizer-master/generateREGOA/NO_plusAlc/pos_sel/P.t. elliotibottom.txt'
    filename3 = '/share/home/liangzhongming/NetOrganizer/RE-ORGANizer-master/generateREGOA/NO_plusAlc/imprinted_gene/Imprinted genes.txt'
    filename4 = '/share/home/liangzhongming/NetOrganizer/RE-ORGANizer-master/generateREGOA/NO_plusAlc/ginbon_gene/PAML_pos_gene.xlsx'
    # 定义列表
    sel_gene_list = []
    cnt_sel_gene_list = 0
    # txt文件
    # with open(filename3) as f:
    #     next(f)
    #     for line in f:
    #         line = line.split()
    #         gene = line[0].upper()
    #         sel_gene_list.append(gene)
    #         cnt_sel_gene_list += 1

    # excel文件
    wb = openpyxl.load_workbook(filename4)
    # 选择第一个工作表
    sheet = wb.active
    for row in sheet.iter_rows(min_row=3, min_col=3, max_col=8):
        sel_gene_list.append(row[0].value)
        cnt_sel_gene_list += 1

    # 2个背景基因
    geneORGANizer = datas.import_database_geneORGANizer()
    grn = datas.import_database_grnhg()
    ppi = datas.importppi() 
    total_geneset = (set(geneORGANizer['genes'].keys()).union(set(ppi.keys()))).union(set(grn['gene_tfs'].keys()))
    vecs = importvec('/share/home/liangzhongming/NetOrganizer/RE-ORGANizer-master/generateREGOA/LiCO/vec/' + '10_15_7path_addGB_4layerTrain_humanGRN_2_vec.txt')

    cnt_bg_gene_list1 = 0
    bg_gene_list1 = []
    for gene in geneORGANizer['genes']:
        if gene in vecs:
            bg_gene_list1.append(gene)
            cnt_bg_gene_list1 += 1

    cnt_bg_gene_list2 = 0
    bg_gene_list2 = []
    for gene in total_geneset:
        if gene in vecs:
            bg_gene_list2.append(gene)
            cnt_bg_gene_list2 += 1

    organ_list = [organ for organ in geneORGANizer['bodyparts']]


    print(f"cnt of sel gene : {cnt_sel_gene_list}, cnt of bg1 gene : {cnt_bg_gene_list1}, cnt of bg2 gene : {cnt_bg_gene_list2}, cnt of organ : {len(organ_list)}")

    return sel_gene_list, bg_gene_list1, bg_gene_list2, organ_list

def echo():

    sel_gene_list, bg_gene_list1, bg_gene_list2, organ_list = handle()


    # 注释信息
    geneORGANizer = datas.import_database_geneORGANizer()
    fmax, bodyparts = getthres()
    vecs = importvec('/share/home/liangzhongming/NetOrganizer/RE-ORGANizer-master/generateREGOA/LiCO/vec/' + '10_15_7path_addGB_4layerTrain_humanGRN_2_vec.txt')

    # 初始化一个空列表，用于存储每个器官的显著性分析结果
    organ_results = []

    # 循环遍历每个器官
    for organ in organ_list:

        # 统计当前器官在两个基因列表中注释的基因数
        # 1.基于原始关系
        # genes_in_organ_1 = [gene for gene in sel_gene_list if gene in geneORGANizer['bodyparts'][organ]]
        # genes_in_organ_2 = [gene for gene in bg_gene_list1 if gene in geneORGANizer['bodyparts'][organ]]

        # 2.基于相似度
        genes_in_organ_1 = [gene for gene in sel_gene_list if gene in vecs and cosine_similarity([vecs[gene], vecs[organ]])[0][1] >= fmax[organ]]
        genes_in_organ_2 = [gene for gene in bg_gene_list2 if gene in vecs and cosine_similarity([vecs[gene], vecs[organ]])[0][1] >= fmax[organ]]

        # 统计两个基因列表中的总基因数和共有基因数
        total_genes = len(set(sel_gene_list + bg_gene_list2))
        intersection_genes = len(set(genes_in_organ_1) & set(genes_in_organ_2))

        # 计算超几何分布的p-value
        p = hypergeom.sf(intersection_genes - 1, total_genes, len(genes_in_organ_1), len(genes_in_organ_2))

        # 将当前器官的分析结果存储到organ_results中
        organ_results.append((organ, p))

    # 将所有p-value进行FDR矫正
    p_values = [result[1] for result in organ_results]
    rejected, adjusted_p_values, _, _ = multipletests(p_values, method='fdr_bh')

    # 初始化一个空列表，用于存储显著富集的器官
    significant_organs = []

    # 找到调整p-value小于0.05的器官
    for i in range(len(organ_results)):
        if adjusted_p_values[i] <= 0.05:
            significant_organs.append(organ_results[i])

    # 将结果按照p-value升序排序
    # organ_results = sorted(organ_results, key=lambda x: x[1])
    significant_organs = sorted(significant_organs, key=lambda x: x[1])

    # 将结果写入本地.txt文件
    with open('echo/bg2_meth2_significant_organs.txt', 'w') as f:
        for organ, p_value in significant_organs:
            f.write("{}\t{}\n".format(organ, p_value))
    
    # 制作富集气泡图
    df = pd.DataFrame(significant_organs[:20], columns=['Organ', 'P-value'])
    df['LogP'] = -1 * df['P-value'].apply(np.log10)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Organ', y='LogP', size='LogP', data=df)

    plt.xticks(rotation=90)
    plt.axhline(-np.log10(0.05), color='r', ls='--')
    plt.title('bg2_meth2 Significantly enriched organs')
    plt.tight_layout()

    plt.savefig('echo/bg2_meth2_enrichment_bubble_plot.png')

def pos_sel():
    sel_gene_list, bg_gene_list1, bg_gene_list2, organ_list = handle()


    # 注释信息
    geneORGANizer = datas.import_database_geneORGANizer()
    fmax, bodyparts = getthres()
    vecs = importvec('/share/home/liangzhongming/NetOrganizer/RE-ORGANizer-master/generateREGOA/LiCO/vec/' + '10_15_7path_addGB_4layerTrain_humanGRN_2_vec.txt')

    # 初始化一个空列表，用于存储每个器官的显著性分析结果
    organ_results = []

    # 循环遍历每个器官
    for organ in organ_list:

        # 统计当前器官在两个基因列表中注释的基因数
        # 1.基于原始关系
        # genes_in_organ_1 = [gene for gene in sel_gene_list if gene in geneORGANizer['bodyparts'][organ]]
        # genes_in_organ_2 = [gene for gene in bg_gene_list1 if gene in geneORGANizer['bodyparts'][organ]]

        # # 2.基于相似度
        genes_in_organ_1 = [gene for gene in sel_gene_list if gene in vecs and cosine_similarity([vecs[gene], vecs[organ]])[0][1] >= fmax[organ]]
        genes_in_organ_2 = [gene for gene in bg_gene_list2 if gene in vecs and cosine_similarity([vecs[gene], vecs[organ]])[0][1] >= fmax[organ]]

        # 统计两个基因列表中的总基因数和共有基因数
        total_genes = len(set(sel_gene_list + bg_gene_list2))
        intersection_genes = len(set(genes_in_organ_1) & set(genes_in_organ_2))

        # 计算超几何分布的p-value
        p = hypergeom.sf(intersection_genes - 1, total_genes, len(genes_in_organ_1), len(genes_in_organ_2))

        # 将当前器官的分析结果存储到organ_results中
        organ_results.append((organ, p))

    # 将所有p-value进行FDR矫正
    p_values = [result[1] for result in organ_results]
    rejected, adjusted_p_values, _, _ = multipletests(p_values, method='fdr_bh')

    # 初始化一个空列表，用于存储显著富集的器官
    significant_organs = []

    # 找到调整p-value小于0.05的器官
    for i in range(len(organ_results)):
        if adjusted_p_values[i] <= 0.05:
            significant_organs.append(organ_results[i])

    # 将结果按照p-value升序排序
    # organ_results = sorted(organ_results, key=lambda x: x[1])
    significant_organs = sorted(significant_organs, key=lambda x: x[1])

    # 将结果写入本地.txt文件
    with open('pos_sel/elliotibottom bg2_meth2_significant_organs.txt', 'w') as f:
        for organ, p_value in significant_organs:
            f.write("{}\t{}\n".format(organ, p_value))
    
    # 制作富集气泡图
    df = pd.DataFrame(significant_organs[:20], columns=['Organ', 'P-value'])
    df['LogP'] = -1 * df['P-value'].apply(np.log10)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Organ', y='LogP', size='LogP', data=df)

    plt.xticks(rotation=90)
    plt.axhline(-np.log10(0.05), color='r', ls='--')
    plt.title('elliotibottom bg2_meth2 Significantly enriched organs')
    plt.tight_layout()

    plt.savefig('pos_sel/elliotibottom bg2_meth2_enrichment_bubble_plot.png')

def imprinted():
    sel_gene_list, bg_gene_list1, bg_gene_list2, organ_list = handle()


    # 注释信息
    geneORGANizer = datas.import_database_geneORGANizer()
    fmax, bodyparts = getthres()
    vecs = importvec('/share/home/liangzhongming/NetOrganizer/RE-ORGANizer-master/generateREGOA/LiCO/vec/' + '10_15_7path_addGB_4layerTrain_humanGRN_2_vec.txt')

    # 初始化一个空列表，用于存储每个器官的显著性分析结果
    organ_results = []

    # 循环遍历每个器官
    for organ in organ_list:

        # 统计当前器官在两个基因列表中注释的基因数
        # 1.基于原始关系
        genes_in_organ_1 = [gene for gene in sel_gene_list if gene in geneORGANizer['bodyparts'][organ]]
        genes_in_organ_2 = [gene for gene in bg_gene_list1 if gene in geneORGANizer['bodyparts'][organ]]

        # # 2.基于相似度
        # genes_in_organ_1 = [gene for gene in sel_gene_list if gene in vecs and cosine_similarity([vecs[gene], vecs[organ]])[0][1] >= fmax[organ]]
        # genes_in_organ_2 = [gene for gene in bg_gene_list1 if gene in vecs and cosine_similarity([vecs[gene], vecs[organ]])[0][1] >= fmax[organ]]

        # 统计两个基因列表中的总基因数和共有基因数
        total_genes = len(set(sel_gene_list + bg_gene_list1))
        intersection_genes = len(set(genes_in_organ_1) & set(genes_in_organ_2))

        # 计算超几何分布的p-value
        p = hypergeom.sf(intersection_genes - 1, total_genes, len(genes_in_organ_1), len(genes_in_organ_2))

        # 将当前器官的分析结果存储到organ_results中
        organ_results.append((organ, p))

    # 将所有p-value进行FDR矫正
    p_values = [result[1] for result in organ_results]
    rejected, adjusted_p_values, _, _ = multipletests(p_values, method='fdr_bh')

    # 初始化一个空列表，用于存储显著富集的器官
    significant_organs = []

    # 找到调整p-value小于0.05的器官
    for i in range(len(organ_results)):
        if adjusted_p_values[i] <= 0.05:
            significant_organs.append(organ_results[i])

    # 将结果按照p-value升序排序
    # organ_results = sorted(organ_results, key=lambda x: x[1])
    significant_organs = sorted(significant_organs, key=lambda x: x[1])

    # 将结果写入本地.txt文件
    with open('imprinted_gene/bg1_meth1_significant_organs.txt', 'w') as f:
        for organ, p_value in significant_organs:
            f.write("{}\t{}\n".format(organ, p_value))
    
    # 制作富集气泡图
    df = pd.DataFrame(significant_organs[:20], columns=['Organ', 'P-value'])
    df['LogP'] = -1 * df['P-value'].apply(np.log10)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Organ', y='LogP', size='LogP', data=df)

    plt.xticks(rotation=90)
    plt.axhline(-np.log10(0.05), color='r', ls='--')
    plt.title('imprinted bg1_meth1 Significantly enriched organs')
    plt.tight_layout()

    plt.savefig('imprinted_gene/bg1_meth1_enrichment_bubble_plot.png')

def monkey():
    sel_gene_list, bg_gene_list1, bg_gene_list2, organ_list = handle()


    # 注释信息
    geneORGANizer = datas.import_database_geneORGANizer()
    fmax, bodyparts = getthres()
    vecs = importvec('/share/home/liangzhongming/NetOrganizer/RE-ORGANizer-master/generateREGOA/LiCO/vec/' + '10_15_7path_addGB_4layerTrain_humanGRN_2_vec.txt')

    # 初始化一个空列表，用于存储每个器官的显著性分析结果
    organ_results = []

    # 循环遍历每个器官
    for organ in organ_list:

        # 统计当前器官在两个基因列表中注释的基因数
        # 1.基于原始关系
        genes_in_organ_1 = [gene for gene in sel_gene_list if gene in geneORGANizer['bodyparts'][organ]]
        genes_in_organ_2 = [gene for gene in bg_gene_list2 if gene in geneORGANizer['bodyparts'][organ]]

        # # 2.基于相似度
        # genes_in_organ_1 = [gene for gene in sel_gene_list if gene in vecs and cosine_similarity([vecs[gene], vecs[organ]])[0][1] >= fmax[organ]]
        # genes_in_organ_2 = [gene for gene in bg_gene_list2 if gene in vecs and cosine_similarity([vecs[gene], vecs[organ]])[0][1] >= fmax[organ]]

        # 统计两个基因列表中的总基因数和共有基因数
        total_genes = len(set(sel_gene_list + bg_gene_list2))
        intersection_genes = len(set(genes_in_organ_2) & set(genes_in_organ_2))

        # 计算富集比
        genes_in_bg = len(genes_in_organ_2)
        if genes_in_bg == 0:
            enrichment_ratio = "inf"
        elif intersection_genes == 0:
            enrichment_ratio = "0"
        else:
            if len(genes_in_organ_1) == 0 or total_genes == 0:
                    enrichment_ratio = "Undefined"
            else:
                enrichment_ratio = round((intersection_genes/genes_in_bg) / (len(genes_in_organ_1) / total_genes), 2)


        # 计算超几何分布的p-value
        p = hypergeom.sf(intersection_genes - 1, total_genes, len(genes_in_organ_1), len(genes_in_organ_2))

        # 将当前器官的分析结果存储到organ_results中
        organ_results.append((organ, p, enrichment_ratio, genes_in_bg))

    # 将所有p-value进行FDR矫正
    p_values = [result[1] for result in organ_results]
    rejected, adjusted_p_values, _, _ = multipletests(p_values, method='fdr_bh')

    # 初始化一个空列表，用于存储显著富集的器官
    significant_organs = []

    # 找到调整p-value小于0.05的器官
    for i in range(len(organ_results)):
        if adjusted_p_values[i] <= 0.05:
            significant_organs.append(organ_results[i])

    # 将结果按照p-value升序排序
    # organ_results = sorted(organ_results, key=lambda x: x[1])
    significant_organs = sorted(significant_organs, key=lambda x: x[1])

    # 将结果写入本地.txt文件
    with open('monkey/bg2_meth1_significant_organs.txt', 'w') as f:
        for organ, p_value, enrichment_ratio in significant_organs:
            f.write("{}\t{}\t{}\n".format(organ, p_value, enrichment_ratio))
    
    # 制作富集气泡图
    df = pd.DataFrame(significant_organs[:20], columns=['Organ', 'P-value', 'Enrichment_ratio'])
    df = df.rename(columns={'Enrichment_ratio': 'Enrichment_ratio'})  # replace space with underscore
    df['LogP'] = -1 * df['P-value'].apply(np.log10)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Organ', y='LogP', size='Enrichment ratio', data=df)

    plt.xticks(rotation=90)
    plt.axhline(-np.log10(0.05), color='r', ls='--')
    plt.title('monkey bg2_meth1 Significantly enriched organs')
    plt.tight_layout()

    plt.savefig('monkey/bg2_meth1_enrichment_bubble_plot.png')

def monkey1():
    sel_gene_list, bg_gene_list1, bg_gene_list2, organ_list = handle()


    # 注释信息
    geneORGANizer = datas.import_database_geneORGANizer()
    fmax, bodyparts = getthres()
    vecs = importvec('/share/home/liangzhongming/NetOrganizer/RE-ORGANizer-master/generateREGOA/LiCO/vec/' + '10_15_7path_addGB_4layerTrain_humanGRN_2_vec.txt')

    # 初始化一个空列表，用于存储每个器官的显著性分析结果
    organ_results = []

    # 循环遍历每个器官
    for organ in organ_list:

        # 统计当前器官在两个基因列表中注释的基因数
        # 1.基于原始关系
        genes_in_organ_1 = [gene for gene in sel_gene_list if gene in geneORGANizer['bodyparts'][organ]]
        genes_in_organ_2 = [gene for gene in bg_gene_list2 if gene in geneORGANizer['bodyparts'][organ]]

        # # 2.基于相似度
        # genes_in_organ_1 = [gene for gene in sel_gene_list if gene in vecs and cosine_similarity([vecs[gene], vecs[organ]])[0][1] >= fmax[organ]]
        # genes_in_organ_2 = [gene for gene in bg_gene_list2 if gene in vecs and cosine_similarity([vecs[gene], vecs[organ]])[0][1] >= fmax[organ]]

        # 统计两个基因列表中的总基因数和共有基因数
        total_genes = len(set(sel_gene_list + bg_gene_list2))
        intersection_genes = len(set(genes_in_organ_2) & set(genes_in_organ_2))

        # 计算富集比和fold enrichment
        genes_in_bg = len(genes_in_organ_2)
        if genes_in_bg == 0:
            enrichment_ratio = "inf"
        elif intersection_genes == 0:
            enrichment_ratio = "0"
        else:
            if len(genes_in_organ_1) == 0 or total_genes == 0:
                    enrichment_ratio = "Undefined"
            else:
                enrichment_ratio = round((intersection_genes/genes_in_bg) / (len(genes_in_organ_1) / total_genes), 2)  # fold enrichment
                enrichment_ratio = round(enrichment_ratio, 2)

        # 计算超几何分布的p-value
        p = hypergeom.sf(intersection_genes - 1, total_genes, len(genes_in_organ_1), len(genes_in_organ_2))

        # 将当前器官的分析结果存储到organ_results中
        organ_results.append((organ, p, enrichment_ratio, genes_in_bg))

    # 将所有p-value进行FDR矫正
    p_values = [result[1] for result in organ_results]
    rejected, adjusted_p_values, _, _ = multipletests(p_values, method='fdr_bh')

    # 初始化一个空列表，用于存储显著富集的器官
    significant_organs = []

    # 找到调整p-value小于0.05的器官
    for i in range(len(organ_results)):
        if adjusted_p_values[i] <= 0.05:
            significant_organs.append(organ_results[i])

    # 将结果按照fold enrichment降序排序
    significant_organs = sorted(significant_organs, key=lambda x: x[2], reverse=True)

    # 将结果写入本地.txt文件
    with open('monkey/bg2_meth1_significant_organs.txt', 'w') as f:
        for organ, p_value, enrichment_ratio, count in significant_organs:
            f.write("{}\t{}\t{}\t{}\n".format(organ, p_value, enrichment_ratio, count))

    # 制作富集气泡图
    df = pd.DataFrame(significant_organs[:20], columns=['Organ', 'P-value', 'Fold_enrichment', 'Count'])
    df = df.rename(columns={'Fold_enrichment': 'Fold enrichment'})  # replace space with underscore

    plt.figure(figsize=(10, 6))
    ax = sns.scatterplot(x='Fold enrichment', y='Organ', size='Count', hue='Fold enrichment', data=df[:20], palette='YlGnBu', sizes=(50, 500))

    # 指定坐标轴范围
    ax.set_xlim(0, max(df['Fold enrichment']) + 1)
    ax.set_ylim(-1, len(df) + 5)

    # 设置坐标轴标签和标题
    ax.set_xlabel('Fold enrichment')
    ax.set_ylabel('Organ')
    ax.set_title('Significantly enriched organs in monkey')

    # 设置水平参考线和图例    
    plt.axvline(1, color='grey', ls='--')
    plt.legend(title='Fold enrichment', loc='upper right', bbox_to_anchor=(1.3, 1))

    plt.tight_layout()

    plt.savefig('monkey/bg2_meth1_enrichment_bubble_plot.png')

if __name__ == '__main__':
    # echo()
    # pos_sel()
    # imprinted()
    monkey1()