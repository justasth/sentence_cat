import numpy as np
import jieba
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, Counter
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import re

# ==================== 语义分类定义 ====================
CLASS_CONFIG = {
    "target": {
        "type": "keyword",
        "dict_path": "target_terms.txt",
        "categories": {
            "经济": [
                "产业发展", "农民增收", "GDP增长", "三产融合", "集体经济", 
                "产业升级", "农村电商", "创业就业", "产值提升", "新型农业经营主体",
                "农产品品牌", "县域经济", "飞地经济", "庭院经济", "直播带货",
                "冷链物流", "农村金融", "三产联动", "消费帮扶", "集体经济股份制改革"
            ],
            "政治": [
                "基层治理", "党建引领", "村民自治", "组织建设", "廉政建设",
                "公共服务", "网格化管理", "平安乡村", "信访维稳", "小微权力清单",
                "四议两公开", "驻村第一书记", "村务监督", "智慧党建", "红色物业",
                "乡贤参事会", "道德评议会", "信用积分制", "党建示范点", "村级事务阳光工程"
            ],
            "生态": [
                "绿水青山", "环境整治", "污染防治", "低碳转型", "生态补偿",
                "垃圾分类", "绿色农业", "可持续发展", "碳中和", "生态产品价值实现",
                "山水林田湖草", "生物多样性保护", "生态廊道", "无废乡村", "生态银行",
                "农药化肥减量", "秸秆综合利用", "生态搬迁", "生态修复工程", "碳汇交易"
            ],
            "社会": [
                "城乡融合", "公共服务均等化", "社会保障", "教育公平", "医疗普惠",
                "住房保障", "精准扶贫", "弱势群体", "收入分配", "社区养老",
                "留守儿童关爱", "残疾人托养", "乡村医生", "文化礼堂", "老年食堂",
                "救急难基金", "微型养老院", "邻里互助", "乡村复兴少年宫", "防返贫监测"
            ],
            "文化": [
                "乡风文明", "传统村落", "文化遗产", "乡土记忆", "非遗保护",
                "文化礼堂", "乡村旅游", "红色文化", "民俗活动", "村史馆",
                "家风家训", "文化示范村", "乡村文创", "非遗工坊", "二十四节气",
                "乡村博物馆", "文化走亲", "非遗传承人", "农耕文化体验", "乡村春晚"
            ]
        }
    },
    "gripper": {
        "type": "keyword",
        "dict_path": "gripper_terms.txt",
        "categories": {
            "人口": [
                "农民工返乡", "人才引进", "户籍改革", "人口迁移", "就业培训",
                "新乡贤", "留守群体", "人口老龄化", "市民化", "银龄人才",
                "乡村CEO", "土专家", "田秀才", "乡村工匠", "农村经纪人",
                "人才驿站", "返乡创业园", "乡愁人才库", "新型职业农民", "乡村规划师"
            ],
            "土地": [
                "宅基地改革", "耕地保护", "土地流转", "增减挂钩", "集体经营性建设用地",
                "土地集约", "占补平衡", "复垦整治", "点状供地", "设施农业用地",
                "土地托管", "农地入市", "宅基地三权分置", "永久基本农田", "土地整治",
                "低效用地再开发", "土地经营权抵押", "闲置宅基地盘活", "全域土地综合整治", "生态用地"
            ],
            "产业": [
                "三产融合", "特色产业集群", "预制菜产业", "中央厨房", "田园综合体",
                "现代农业产业园", "乡村旅游民宿", "农村电商", "直播助农", "冷链物流",
                "绿色有机农业", "林下经济", "庭院经济", "认养农业", "共享农庄",
                "农业产业化联合体", "村企联建", "飞地经济", "乡村文旅IP", "非遗工坊",
                "数字农业", "农业众筹", "农产品区域公用品牌", "地理标志产品", "生态循环农业",
                "农业社会化服务", "农机共享", "农业全产业链", "产业强镇", "一县一业"
            ],
            "资金": [
                "财政补贴", "社会资本", "金融支农", "PPP模式", "产业基金",
                "信贷支持", "投资引导", "股权合作", "惠农贷款", "乡村振兴债",
                "农业保险", "融资担保", "数字普惠金融", "供应链金融", "绿色金融",
                "消费扶贫", "公益众筹", "碳金融", "数字人民币试点", "农村产权抵押"
            ],
            "科创": [
                "数字乡村", "智慧农业", "物联网", "大数据", "电商平台",
                "机械化", "科技特派员", "技术培训", "智能装备", "区块链",
                "农业无人机", "遥感监测", "智能灌溉", "数字田园", "农产品溯源","科创","科技","技术改造","产品研发",
                "5G应用", "农业机器人", "生物育种", "智慧农业物联网平台", "冷链保鲜技术","专家智库","科技成果转化应用"
            ],
            "党建": [
                "合作社", "家庭农场", "村集体公司", "农业龙头企业", "社会化服务组织",
                "村民理事会", "乡贤理事会", "联合体", "农民合作社联合社", "产业联盟",
                "村投公司", "强村公司", "党建联盟", "共富工坊", "共享食堂",
                "红白理事会", "劳务合作社", "土地股份合作社", "乡村运营团队", "区域党建联合体",
                "三权分置", "产权改革", "生态补偿机制", "碳交易", "村民积分制",
                "考核评价", "容错机制", "试点示范", "政策包", "三变改革",
                "标准地改革", "亩均论英雄", "河长制", "林长制", "田长制","机构改革",
                "生态产品GEP核算", "宅基地资格权", "集体资产股权质押", "农业标准体系", "乡村治理积分制"
            ],
            "民生": [
                "厕所革命", "危房改造", "饮水安全", "四好农村路", "农村电网升级",
                "农村养老驿站", "乡村幸福院", "医养结合", "送药上山", "城乡教共体",
                "营养午餐计划", "乡村文化礼堂", "农民夜校", "农村电影放映", "助残公益岗",
                "农民工欠薪治理", "留守儿童关爱", "乡村互助基金", "村级卫生室标准化",
                "快递进村", "电力普遍服务", "农村厕所管护", "微型消防站", "应急广播","教育","医疗"
            ]
        }
    },
    "mechanism": {
        "type": "keyword",
        "dict_path": "_terms.txt",
        "categories": {
            "供给型": [
                "财政拨款", "专项资金", "转移支付", "国债资金", 
                "道路硬化", "电网改造", "水利工程", "通信基站", "物流网点",
                "技能培训", "职业农民", "专家下乡", "人才公寓",
                "智慧农业", "数字平台", "智能装备", "遥感监测",
                "垃圾处理", "污水治理", "厕所革命", "村容提升",
                "土地平整", "村庄规划", "集中居住", "功能分区"
            ],
            "环境型": [
                "五年规划", "行动方案", "发展纲要", "专项规划",
                "立法保障", "执法监督", "标准体系", "合规审查",
                "联席会议", "领导小组", "专班推进", "包保责任",
                "负面清单", "容错机制", "奖惩制度", "考核指标",
                "监测平台", "评估机制", "信用体系", "追溯系统",
                "试点工程", "改革授权", "容缺受理", "首错免责",
                "以奖代补", "贷款贴息", "风险补偿", "担保基金"
            ],
            "需求型": [
                "PPP模式", "产业基金", "股权合作", "众筹融资",
                "乡贤回归", "村企共建", "志愿帮扶", "公益捐赠",
                "产权交易", "价格形成", "准入清单", "竞争中性",
                "示范基地", "标杆项目", "典型培育", "样板打造"
            ]
        }
    }
}

class PolicyAnalyzer:
    def __init__(self):
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.classifiers = {k: None for k in CLASS_CONFIG.keys()}
        self.vectorizers = {k: TfidfVectorizer(max_features=5000, ngram_range=(1, 2)) for k in CLASS_CONFIG.keys()}
        self.label_encoders = {k: LabelEncoder() for k in CLASS_CONFIG.keys()}
        self.reset_data()

    def reset_data(self):
        self.sentence_data = []
        self.word_freq = {k: defaultdict(int) for k in CLASS_CONFIG.keys()}

    def split_sentences(self, text):
        if not text:
            return []
        
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'([。！？；])\s*\n+\s*([^。！？；])', r'\1\2', text)
        sentence_endings = r'([。！？])'
        raw_splits = re.split(sentence_endings, text)
        
        sentences = []
        i = 0
        current_sentence = ""
        
        while i < len(raw_splits):
            if i + 1 < len(raw_splits):
                current_sentence += raw_splits[i] + raw_splits[i+1]
                if len(current_sentence.strip()) >= 20:
                    sentences.append(current_sentence.strip())
                    current_sentence = ""
                i += 2
            else:
                current_sentence += raw_splits[i]
                if len(current_sentence.strip()) >= 10:
                    sentences.append(current_sentence.strip())
                elif sentences and current_sentence.strip():
                    sentences[-1] = sentences[-1] + current_sentence.strip()
                i += 1
        
        return [s for s in sentences if '乡' in s or '农' in s]

    def calculate_similarity(self, text1, text2):
        def tokenize(text):
            return [w for w in jieba.cut(text) if len(w.strip()) > 0]

        def jaccard_sim(tokens1, tokens2):
            set1, set2 = set(tokens1), set(tokens2)
            return len(set1 & set2) / len(set1 | set2) if set1 | set2 else 0.0

        def tfidf_sim(tokens1, tokens2):
            vocab = list(set(tokens1 + tokens2))
            word_to_idx = {word: i for i, word in enumerate(vocab)}
            
            def get_vector(tokens):
                vector = [0] * len(vocab)
                for word, count in Counter(tokens).items():
                    if word in word_to_idx:
                        vector[word_to_idx[word]] = count
                return vector
            
            vec1, vec2 = get_vector(tokens1), get_vector(tokens2)
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            norm1, norm2 = sum(a * a for a in vec1) ** 0.5, sum(b * b for b in vec2) ** 0.5
            return dot_product / (norm1 * norm2) if norm1 and norm2 else 0.0

        tokens1, tokens2 = tokenize(text1), tokenize(text2)
        if not tokens1 or not tokens2:
            return 0.0
            
        return 0.6 * jaccard_sim(tokens1, tokens2) + 0.4 * tfidf_sim(tokens1, tokens2)

    def _classify(self, sentence, system_name):
        if hasattr(self, 'train_data'):
            best_match = None
            max_similarity = 0
            
            for _, row in self.train_data.iterrows():
                similarity = self.calculate_similarity(sentence, str(row['content']))
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match = row
                    if similarity >= 0.85:
                        return {"category": row[f'{system_name}_cat'], "confidence": similarity}
            
            if max_similarity >= 0.8:
                return {"category": best_match[f'{system_name}_cat'], "confidence": max_similarity}
        
        if self.classifiers[system_name] is not None:
            try:
                X = self.vectorizers[system_name].transform([sentence])
                y_pred = self.classifiers[system_name].predict(X)
                y_pred_proba = self.classifiers[system_name].predict_proba(X)
                return {
                    "category": self.label_encoders[system_name].inverse_transform(y_pred)[0],
                    "confidence": float(max(y_pred_proba[0]))
                }
            except Exception:
                pass
        
        return self._semantic_classify(sentence, system_name)

    def _semantic_classify(self, sentence, system_name):
        if system_name in CLASS_CONFIG and "categories" in CLASS_CONFIG[system_name]:
            for category, keywords in CLASS_CONFIG[system_name]["categories"].items():
                if any(keyword.lower() in sentence.lower() for keyword in keywords):
                    return {"category": category, "confidence": 1.0}
        
        embeddings = self.category_systems[system_name]["embeddings"]
        sentence_embedding = self.model.encode([sentence])
        similarities = cosine_similarity(sentence_embedding, embeddings)[0]
        max_index = np.argmax(similarities)
        
        return {
            "category": self.category_systems[system_name]["categories"][max_index],
            "confidence": float(similarities[max_index])
        }

    def analyze(self, text):
        if not text:
            return
        
        sentences = self.split_sentences(text)
        if not sentences:
            return
        
        for idx, sentence in enumerate(sentences, 1):
            classification = {"sentence_id": idx, "content": sentence}
            
            for system_name in CLASS_CONFIG.keys():
                result = self._classify(sentence, system_name)
                classification[f"{system_name}_cat"] = result["category"]
                classification[f"{system_name}_conf"] = result["confidence"]
                
                for word in [w for w in jieba.lcut(sentence) if len(w) > 1 and not w.isdigit()]:
                    self.word_freq[system_name][word] += 1
            
            self.sentence_data.append(classification)

    def get_results(self):
        return {
            "sentences": pd.DataFrame(self.sentence_data),
            "word_frequencies": self.word_freq
        }
