import os
import random
import shutil
from random import randint

gO_organs = ['cerebellum','brain','head','uterus','ear','outer ear','middle ear','inner ear','peripheral nerve','peripheral nervous system','cranial nerve','pituitary gland','ovary','testicle','penis','vagina','vulva','skin','hair','prostate','breast','eye','lower limb','spinal cord','mouth','skull','spinal column','vertebrae','pelvis','rib','rib cage','lung','foot','tibia','fibula','femur','sternum','chest wall','clavicle','scapula','humerus','radius','ulna','neck','mandible','maxilla','jaw','ankle','knee','hip','wrist','elbow','shoulder','finger','toe','digit','hand','forearm','arm','shin','thigh','upper limb','face','eyelid','kidney','urethra','heart','anus','urinary bladder','blood vessel','pancreas','bronchus','white blood cell','blood','liver','biliary tract','gallbladder','olfactory bulb','ureter','red blood cell','coagulation system','aorta','heart valve','rectum','tooth','nose','intestine','large intestine','forehead','diaphragm','stomach','small intestine','chin','cheek','bone marrow','lip','pharynx','nail','meninges','cerebrospinal fluid','spleen','lymph node','sinus','abdominal wall','duodenum','esophagus','placenta','larynx','trachea','vocal cord','epiglottis','tongue','scalp','lymph vessel','lacrimal apparatus','adrenal gland','sweat gland','salivary gland','thyroid','hypothalamus','appendix','parathyroid','thymus','fallopian tube','vas deferens']
gO_regions = ['limbs', 'head and neck', 'thorax', 'abdomen', 'pelvis', 'general']
gO_systems = ['immune',	'cardiovascular',	'nervous',	'skeleton',	'skeletal muscle',	'reproductive',	'digestive',	'urinary',	'respiratory',	'endocrine',	'lymphatic',	'integumentary']
gO_germlayers = ['endoderm', 'mesoderm', 'ectoderm']

# gO_totalLayers = ['cerebellum','brain','head','uterus','ear','outer ear','middle ear','inner ear','peripheral nerve','peripheral nervous system','cranial nerve','pituitary gland','ovary','testicle','penis','vagina','vulva','skin','hair','prostate','breast','eye','lower limb','spinal cord','mouth','skull','spinal column','vertebrae','pelvis','rib','rib cage','lung','foot','tibia','fibula','femur','sternum','chest wall','clavicle','scapula','humerus','radius','ulna','neck','mandible','maxilla','jaw','ankle','knee','hip','wrist','elbow','shoulder','finger','toe','digit','hand','forearm','arm','shin','thigh','upper limb','face','eyelid','kidney','urethra','heart','anus','urinary bladder','blood vessel','pancreas','bronchus','white blood cell','blood','liver','biliary tract','gallbladder','olfactory bulb','ureter','red blood cell','coagulation system','aorta','heart valve','rectum','tooth','nose','intestine','large intestine','forehead','diaphragm','stomach','small intestine','chin','cheek','bone marrow','lip','pharynx','nail','meninges','cerebrospinal fluid','spleen','lymph node','sinus','abdominal wall','duodenum','esophagus','placenta','larynx','trachea','vocal cord','epiglottis','tongue','scalp','lymph vessel','lacrimal apparatus','adrenal gland','sweat gland','salivary gland','thyroid','hypothalamus','appendix','parathyroid','thymus','fallopian tube','vas deferens','immune', 'cardiovascular', 'nervous', 'skeleton', 'skeletal', 'muscle', 'reproductive', 'digestive',	'urinary', 'respiratory', 'endocrine', 'lymphatic', 'integumentary','limbs', 'head and neck', 'thorax', 'abdomen', 'pelvis', 'general','endoderm', 'mesoderm', 'ectoderm']
gO_totalLayers= ['cerebellum','brain','head','uterus','ear','outer ear','middle ear','inner ear','peripheral nerve','peripheral nervous system','cranial nerve','pituitary gland','ovary','testicle','penis','vagina','vulva','skin','hair','prostate','breast','eye','lower limb','spinal cord','mouth','skull','spinal column','vertebrae','pelvis','rib','rib cage','lung','foot','tibia','fibula','femur','sternum','chest wall','clavicle','scapula','humerus','radius','ulna','neck','mandible','maxilla','jaw','ankle','knee','hip','wrist','elbow','shoulder','finger','toe','digit','hand','forearm','arm','shin','thigh','upper limb','face','eyelid','kidney','urethra','heart','anus','urinary bladder','blood vessel','pancreas','bronchus','white blood cell','blood','liver','biliary tract','gallbladder','olfactory bulb','ureter','red blood cell','coagulation system','aorta','heart valve','rectum','tooth','nose','intestine','large intestine','forehead','diaphragm','stomach','small intestine','chin','cheek','bone marrow','lip','pharynx','nail','meninges','cerebrospinal fluid','spleen','lymph node','sinus','abdominal wall','duodenum','esophagus','placenta','larynx','trachea','vocal cord','epiglottis','tongue','scalp','lymph vessel','lacrimal apparatus','adrenal gland','sweat gland','salivary gland','thyroid','hypothalamus','appendix','parathyroid','thymus','fallopian tube','vas deferens','immune','cardiovascular','nervous','skeleton','skeletal muscle','reproductive','digestive','urinary','respiratory','endocrine','lymphatic','integumentary','limbs', 'head and neck', 'thorax', 'abdomen', 'pelvis', 'general','endoderm', 'mesoderm', 'ectoderm']

def generate_pos_neg(filename='./model0.87/data/DX_total4Layers.txt'):

    # 生成neg.txt和pos.txt
    # fneg = open('./model0.87/data/DX_neg.txt', 'w')
    # count = 0
    # with open(filename, 'r') as f:
    #     for line in f:
    #         line = line.split()
    #         gene = line[0]
    #         # print(len(line))
    #         for index in range(1, 147):  # 125 + 12 + 6 + 3 = 146
    #             # print('index: ' + str(index))
    #             # print('line[index]: ' + str(line[index]))
    #             if line[index] == '0':
    #                 count += 1
    #                 # print(gene, ' count: ', count)
    #         for index in range(1, 147):
    #             if count == 146:  # 该基因与任何bodypart都无关,不考虑
    #                 with open('./model0.87/data/abandon_gene.txt', 'a+') as fa:
    #                     fa.write(str(gene) + '\n')
    #                     count = 0
    #                 break
    #             else:
    #                 if line[index] == '0':  # 构造负样本
    #                     bodyPart = gO_totalLayers[index - 1]
    #                     fneg.write(gene + '\t')
    #                     fneg.write(bodyPart + '\n')
    #                 count = 0

    fpos = open('./model0.87/data/DX_pos.txt', 'w')
    count = 0
    with open(filename, 'r') as f:
        for line in f:
            line = line.split()
            gene = line[0]
            # print(len(line))
            for index in range(1, 147):  # 125 + 12 + 6 + 3 = 146
                # print('index: ' + str(index))
                # print('line[index]: ' + str(line[index]))
                if line[index] == '0':
                    count += 1
                    # print(gene, ' count: ', count)
            for index in range(1, 147):
                if count == 146:  # 该基因与任何bodypart都无关,不考虑
                    with open('./model0.87/data/abandon_gene.txt', 'a+') as fa:
                        fa.write(str(gene) + '\n')
                        count = 0
                    break
                else:
                    if line[index] == '1':  # 构造正样本
                        bodyPart = gO_totalLayers[index - 1]
                        fpos.write(gene + '\t')
                        fpos.write(bodyPart + '\n')
                    count = 0

    # fpos = open('./model0.87/data/new_gene_systemName.txt', 'w')
    # with open(filename, 'r') as f:
    #     for line in f:
    #         line = line.split()
    #         gene = line[0]
    #         # print(len(line))
    #         for index in range(1, 13):
    #             # print('index: ' + str(index))
    #             # print('line[index]: ' + str(line[index]))
    #             if line[index] == '1':  # 构造正样本
    #                 bodyPart = gO_systems[index - 1]
    #                 fpos.write(gene + '\t')
    #                 fpos.write(bodyPart + '\n')


def generate_4trainSet():
    #
    oldf = open('./process_result/pos.txt', 'r', encoding='utf-8')
    trainf = open('./process_result/train5.txt', 'w', encoding='utf-8')
    testf = open('./process_result/test5.txt', 'w', encoding='utf-8')
    resultList = random.sample(range(0, 167895), 134318)  # 8:2
    # resultList = random.sample(range(0, 167895), 125921)  # 75:25

    lines = oldf.readlines()
    for i in resultList:
        trainf.write(lines[i])

    for i in range(0, 167895):
        if i not in resultList:
            testf.write(lines[i])

    oldf.close()
    trainf.close()
    testf.close()


def generate_4layer_4trainSet():
    #
    oldf = open('./model0.87/data/new_gene_systemName.txt', 'r', encoding='utf-8')
    trainf = open('./model0.87/data/train_new_gene_systemName.txt', 'w', encoding='utf-8')
    testf = open('./model0.87/data/test_new_gene_systemName.txt', 'w', encoding='utf-8')
    resultList = random.sample(range(0, 23042), 18434)  # 8:2  9121 * 0.8 = 7295  120330 * 0.8 = 96264  15401 * 0.8 = 12320  23043 * 0.8 = 18434
    # resultList = random.sample(range(0, 167895), 125921)  # 75:25

    lines = oldf.readlines()
    for i in resultList:
        trainf.write(lines[i])

    for i in range(0, 23042):
        if i not in resultList:
            testf.write(lines[i])

    oldf.close()
    trainf.close()
    testf.close()


def lzm_test():
    oldf = open('./process_result/lzm_pos.txt', 'r', encoding='utf-8')
    trainf = open('./process_result/lzm_train.txt', 'w+', encoding='utf-8')
    testf = open('./process_result/lzm_test.txt', 'w+', encoding='utf-8')
    resultList = random.sample(range(0, 12), 10)  # 8:2

    lines = oldf.readlines()
    print(lines)
    for i in resultList:
        print(i)
        trainf.write(lines[i])

    for i in range(0, 12):
        if i not in resultList:
            print(i)
            testf.write(lines[i])

    oldf.close()
    trainf.close()
    testf.close()

if __name__=='__main__':

    # generate_4trainSet()
    # lzm_test()
    generate_pos_neg()
    # generate_train_test()
    # generate_4layer_4trainSet()