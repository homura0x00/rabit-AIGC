import os
import json
from py2neo import Graph, Node


class MedicalGraph:
    # 连接数据库
    def __init__(self):
        # 获取当前绝对路径的上层目录
        cur_dir = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        # 获取保存医学数据集的json文件
        self.data_path = os.path.join(cur_dir, './Resources/data/medical.json')
        # 存储图谱的数据库
        self.g = Graph(
            host='127.0.0.1',
            user='neo4j',
            password='123456')

    # 创建实体、属性以及实体间的关系
    def read_nodes(self):
        # 实体
        drugs = []  # 药品
        foods = []  # 食品
        checks = []  # 检查
        departments = []  # 科室
        producers = []  # 药品大类
        diseases = []  # 疾病
        symptoms = []  # 症状

        # 疾病属性，不构建节点，用于描述疾病的信息属性
        disease_infos = []  # 疾病信息

        # 构建实体（节点）间的关系，共11类
        rels_department = []  # 科室 —— 科室
        rels_noteat = []  # 疾病 —— 忌吃食品
        rels_doeat = []  # 疾病 —— 宜吃食品
        rels_recommandeat = []  # 疾病 —— 推荐食物
        rels_commonddrug = []  # 疾病 —— 通用药品
        rels_recommanddrug = []  # 疾病 —— 热门药品
        rels_check = []  # 疾病 —— 检查
        rels_drug_producer = []  # 厂商 —— 药物
        rels_symptom = []  # 疾病的症状
        rels_acompany = []  # 疾病的并发
        rels_category = []  # 疾病与科室之间的关系

        count = 0
        for data in open(self.data_path, encoding='utf-8'):     # 提取json文件内容中每一字段（疾病）的标签内容
            disease_dict = {}
            count += 1
            print(count)
            data_json = json.loads(data)    # 读取数据
            disease = data_json['name']     # 疾病名称
            disease_dict['name'] = disease
            diseases.append(disease)
            disease_dict['desc'] = ''               # 疾病的描述
            disease_dict['prevent'] = ''            # 疾病预防，即如何预防该疾病
            disease_dict['cause'] = ''              # 确诊原因
            disease_dict['easy_get'] = ''           # 易感人群
            disease_dict['cure_department'] = ''    # 治疗科室
            disease_dict['cure_way'] = ''           # 治疗途径
            disease_dict['cure_lasttime'] = ''      # 治疗周期，即康复时间
            disease_dict['symptom'] = ''            # 疾病
            disease_dict['cured_prob'] = ''         # 康复的概率，即能否根治

            # 查找的医学词条是否在正在提取出来的文件中，
            if 'symptom' in data_json:
                symptoms += data_json['symptom']    # 存储症状数据
                # 一个疾病可能带有多个症状，用for循环关联其关系
                for symptom in data_json['symptom']:
                    # 对于每个症状都建立一个“疾病——疾病”的关系
                    rels_symptom.append([disease, symptom])

            if 'acompany' in data_json:
                # 一个疾病可能伴随有多个并发症，用for循环关联其关系
                for acompany in data_json['acompany']:
                    # 建立一个“疾病——伴随疾病”的关系
                    rels_acompany.append([disease, acompany])

            if 'desc' in data_json:
                # 疾病描述，这里不是关系，而是疾病的属性
                disease_dict['desc'] = data_json['desc']

            if 'prevent' in data_json:
                disease_dict['prevent'] = data_json['prevent']  # 疾病预防

            if 'cause' in data_json:
                disease_dict['cause'] = data_json['cause']      # 引起疾病的原因

            if 'get_prob' in data_json:
                disease_dict['get_prob'] = data_json['get_prob']    # 发病率

            if 'easy_get' in data_json:
                disease_dict['easy_get'] = data_json['easy_get']    # 易感人群

            if 'cure_department' in data_json:
                cure_department = data_json['cure_department']      # 所属科室
                if len(cure_department) == 1:
                    rels_category.append(
                        [disease, cure_department[0]])     # 当只对应一个科室的情况
                if len(cure_department) == 2:
                    big = cure_department[0]        # 大科室 A干支
                    small = cure_department[1]      # 小科室  A干支下的B干支
                    rels_department.append([small, big])        # 建立“科室——科室”的关系
                    rels_category.append([disease, small])      # 建立“疾病——科室”的关系

                disease_dict['cure_department'] = cure_department
                departments += cure_department

            if 'cure_way' in data_json:
                disease_dict['cure_way'] = data_json['cure_way']    # 所对应的治疗途径

            if 'cure_lasttime' in data_json:
                # 治疗时间
                disease_dict['cure_lasttime'] = data_json['cure_lasttime']

            if 'cured_prob' in data_json:
                # 康复的概率（根治疾病的成功概率）
                disease_dict['cured_prob'] = data_json['cured_prob']

            if 'common_drug' in data_json:
                common_drug = data_json['common_drug']      # 常用药物（治疗该疾病的药物）
                for drug in common_drug:
                    rels_commonddrug.append([disease, drug])    # 建立“疾病——药物”关系
                drugs += common_drug

            if 'recommand_drug' in data_json:
                # 推荐药物（特效药？个人定制）
                recommand_drug = data_json['recommand_drug']
                drugs += recommand_drug
                for drug in recommand_drug:
                    rels_recommanddrug.append(
                        [disease, drug])      # 建立“疾病——推荐药物”关系

            if 'not_eat' in data_json:
                not_eat = data_json['not_eat']      # 不可食用的食物（A疾病下不可食用）
                for _not in not_eat:
                    rels_noteat.append([disease, _not])     # 建立“疾病——不可使用的食物”关系

                foods += not_eat
                do_eat = data_json['do_eat']        # 可食用的食物
                for _do in do_eat:
                    rels_doeat.append([disease, _do])   # 建立“疾病——可食用的食物”关系

                foods += do_eat
                # 推荐（食用）的食物（加快康复时间）
                recommand_eat = data_json['recommand_eat']

                for _recommand in recommand_eat:
                    rels_recommandeat.append(
                        [disease, _recommand])     # 建立“疾病——推荐的食物”关系
                foods += recommand_eat

            if 'check' in data_json:
                check = data_json['check']      # 检查。一个疾病所需要做的化验
                for _check in check:
                    rels_check.append([disease, _check])    # 建立“疾病——化验”的关系
                checks += check
            if 'drug_detail' in data_json:
                drug_detail = data_json['drug_detail']      # 药物的详细信息
                producer = [i.split('(')[0] for i in drug_detail]
                rels_drug_producer += [[i.split('(')[0], i.split(
                    '(')[-1].replace(')', '')] for i in drug_detail]
                producers += producer
            disease_infos.append(disease_dict)      # 添加疾病的信息list
        return set(drugs), set(foods), set(checks), set(departments), set(producers), set(symptoms), set(diseases), \
            disease_infos, rels_check, rels_recommandeat, rels_noteat, rels_doeat, rels_department, rels_commonddrug, \
            rels_drug_producer, rels_recommanddrug, rels_symptom, rels_acompany, rels_category

    '''建立节点'''

    def create_node(self, label, nodes):
        count = 0
        for node_name in nodes:
            node = Node(label, name=node_name)
            self.g.create(node)
            count += 1
            print(count, len(nodes))
        return

    '''创建知识图谱中心疾病的节点'''

    def create_diseases_nodes(self, disease_infos):
        count = 0
        for disease_dict in disease_infos:
            node = Node("Disease", name=disease_dict['name'], desc=disease_dict['desc'],
                        prevent=disease_dict['prevent'], cause=disease_dict['cause'],
                        easy_get=disease_dict['easy_get'], cure_lasttime=disease_dict['cure_lasttime'],
                        cure_department=disease_dict['cure_department'],
                        cure_way=disease_dict['cure_way'], cured_prob=disease_dict['cured_prob'])   # 各个疾病节点的属性
            self.g.create(node)
            count += 1
            print(count)
        return

    '''创建知识图谱实体节点类型schema'''

    def create_graphnodes(self):
        Drugs, Foods, Checks, Departments, Producers, Symptoms, Diseases, \
            disease_infos, rels_check, rels_recommandeat, rels_noteat, rels_doeat, \
            rels_department, rels_commonddrug, rels_drug_producer, \
            rels_recommanddrug, rels_symptom, rels_acompany, \
            rels_category = self.read_nodes()
        self.create_diseases_nodes(disease_infos)
        self.create_node('Drug', Drugs)     # 创建药物节点
        print(len(Drugs))
        self.create_node('Food', Foods)     # 创建食物节点
        print(len(Foods))
        self.create_node('Check', Checks)   # 创建检查（化验）节点
        print(len(Checks))
        self.create_node('Department', Departments)     # 创建（所属）科室节点
        print(len(Departments))
        self.create_node('Producer', Producers)         # 创建制药厂（药物厂家）节点
        print(len(Producers))
        self.create_node('Symptom', Symptoms)           # 创建症状节点
        return

    '''创建实体关系（边）'''

    def create_graphrels(self):
        Drugs, Foods, Checks, Departments, Producers, \
            Symptoms, Diseases, disease_infos, rels_check, \
            rels_recommandeat, rels_noteat, rels_doeat, \
            rels_department, rels_commonddrug, rels_drug_producer, \
            rels_recommanddrug, rels_symptom, rels_acompany, \
            rels_category = self.read_nodes()
        self.create_relationship(
            'Disease', 'Food', rels_recommandeat, 'recommand_eat', '推荐食谱')
        self.create_relationship(
            'Disease', 'Food', rels_noteat, 'no_eat', '忌吃')
        self.create_relationship('Disease', 'Food', rels_doeat, 'do_eat', '宜吃')
        self.create_relationship(
            'Department', 'Department', rels_department, 'belongs_to', '属于')
        self.create_relationship(
            'Disease', 'Drug', rels_commonddrug, 'common_drug', '常用药品')
        self.create_relationship(
            'Producer', 'Drug', rels_drug_producer, 'drugs_of', '生产药品')
        self.create_relationship(
            'Disease', 'Drug', rels_recommanddrug, 'recommand_drug', '好评药品')
        self.create_relationship(
            'Disease', 'Check', rels_check, 'need_check', '诊断检查')
        self.create_relationship('Disease', 'Symptom',
                                 rels_symptom, 'has_symptom', '症状')
        self.create_relationship('Disease', 'Disease',
                                 rels_acompany, 'acompany_with', '并发症')
        self.create_relationship(
            'Disease', 'Department', rels_category, 'belongs_to', '所属科室')

    '''创建实体关联边'''

    def create_relationship(self, start_node, end_node, edges, rel_type, rel_name):
        count = 0
        # 去重处理
        set_edges = []
        for edge in edges:
            set_edges.append('###'.join(edge))
        all = len(set(set_edges))
        for edge in set(set_edges):
            edge = edge.split('###')
            p = edge[0]
            q = edge[1]
            query = "match(p:%s),(q:%s) where p.name='%s'and q.name='%s' create (p)-[rel:%s{name:'%s'}]->(q)" % (
                start_node, end_node, p, q, rel_type, rel_name)
            try:
                self.g.run(query)
                count += 1
                print(rel_type, count, all)
            except Exception as e:
                print(e)
        return

    '''导出数据'''

    def export_data(self):
        Drugs, Foods, Checks, Departments, Producers, \
            Symptoms, Diseases, disease_infos, rels_check, \
            rels_recommandeat, rels_noteat, rels_doeat, \
            rels_department, rels_commonddrug, rels_drug_producer, \
            rels_recommanddrug, rels_symptom, rels_acompany, \
            rels_category = self.read_nodes()
        f_drug = open('./Resources/dict/drug.txt', 'w+')
        f_food = open('./Resources/dict/food.txt', 'w+')
        f_check = open('./Resources/dict/check.txt', 'w+')
        f_department = open('./Resources/dict/department.txt', 'w+')
        f_producer = open('./Resources/dict/producer.txt', 'w+')
        f_symptom = open('./Resources/dict/symptom.txt', 'w+')
        f_disease = open('./Resources/dict/disease.txt', 'w+')

        f_drug.write('\n'.join(list(Drugs)))
        f_food.write('\n'.join(list(Foods)))
        f_check.write('\n'.join(list(Checks)))
        f_department.write('\n'.join(list(Departments)))
        f_producer.write('\n'.join(list(Producers)))
        f_symptom.write('\n'.join(list(Symptoms)))
        f_disease.write('\n'.join(list(Diseases)))

        f_drug.close()
        f_food.close()
        f_check.close()
        f_producer.close()
        f_symptom.close()
        f_disease.close()

        return


if __name__ == '__main__':
    handler = MedicalGraph()
    print("step1:开始构建图谱实体节点")
    handler.create_graphnodes()
    print("step1:实体节点构建完成!")
    print("step2:开始构建图谱实体间的关系")
    handler.create_graphrels()
    handler.export_data()
    print("step2:实体间的关系构建完成！")
