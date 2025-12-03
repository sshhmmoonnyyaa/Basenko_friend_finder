import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pymorphy2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import warnings
from collections import Counter

warnings.filterwarnings('ignore')

class BERTProcessor:
    def __init__(self):
        self.morph = None
        self.stop_words = None
        self.model = None
        self.df = None
        self.embeddings = None
        self.similarity_matrix = None 
        self.kmeans = None
        self.pca = None
        self.clusters = None 
        
    def initialize_nltk(self):
        try:
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt', quiet=True)
        except:
            pass
    
    def initialize_tools(self):
        self.morph = pymorphy2.MorphAnalyzer()
        
        nltk_stop_words = set(stopwords.words('russian'))
        custom_stop_words = {
            'привет', 'меня', 'звать', 'здравствуйте', 'приветик', 'здарова', 'хай',
            'это', 'вот', 'ну', 'да', 'нет', 'так', 'еще', 'уже', 'просто', 'очень',
            'свой', 'моя', 'мой', 'мое', 'работаю', 'работать', 'своя', 'свои', 'своей',
            'свою', 'своих', 'который', 'которая', 'которые', 'которым', 'которыми',
            'любить','нравится','хотеть','уметь','слушать','искать','заниматься',
            'смотреть', 'обожать', 'девушка','парень','человек', 'фильм','музыка',
            'здорово', 'фанат','работа','жизнь', 'реалистичный', 'фанат',
            'мужчина','увлекаться','любимый','изучать','хобби', 'женщина',
            'мечтать','весь', 'создавать', 'коллекционировать','специалист','время',
            'помогать','сериал', 'создание','классический','система', 'свободный','умный',
            'звук','городской', 'ценить', 'искусство', 'история', 'исторический', 'оценить',
            'разрабатывать','старинный','ребёнок', 'редкий', 'разный', 'музей',
            'мастер','древний','традиционный', 'возвращать','красота','встреча',
            'коллекциониий','коллекционирование', 'изучение', 'год'
        }
        self.stop_words = nltk_stop_words.union(custom_stop_words)
    
    def preprocess_text(self, text):
        if not isinstance(text, str) or not text.strip():
            return ""
        
        text = text.lower()
        text = re.sub(r'[^а-яё\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        tokens = word_tokenize(text, language='russian')
        
        cleaned_tokens = []
        for token in tokens:
            if token not in self.stop_words and len(token) > 2:
                parsed = self.morph.parse(token)[0]
                lemma = parsed.normal_form
                if lemma not in self.stop_words and len(lemma) > 2:
                    cleaned_tokens.append(lemma)
                    
        return " ".join(cleaned_tokens)

    def load_and_clean_data(self, excel_path='base_doc.xlsx'):
        self.df = pd.read_excel(excel_path)
        self.df = self.df.dropna(subset=['Описание'])
        self.df = self.df[self.df['Описание'].str.strip() != '']
        
        self.initialize_nltk()
        self.initialize_tools()
        
        self.df['processed_text'] = self.df['Описание'].apply(self.preprocess_text)
        return self.df

    def create_bert_embeddings(self):
        self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        self.embeddings = self.model.encode(self.df['processed_text'].tolist(), show_progress_bar=True)
        self.similarity_matrix = cosine_similarity(self.embeddings) 
        return self.embeddings

    def perform_clustering(self, n_clusters=6):
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.clusters = self.kmeans.fit_predict(self.embeddings)
        self.df['cluster'] = self.clusters
        
        self.pca = PCA(n_components=2)
        vectors_2d = self.pca.fit_transform(self.embeddings)
        self.df['pca_x'] = vectors_2d[:, 0]
        self.df['pca_y'] = vectors_2d[:, 1]
        
        return self.clusters

    def get_cluster_info(self, cluster_id):
        cluster_data = self.df[self.df['cluster'] == cluster_id]
        if len(cluster_data) == 0:
            return {'size': 0, 'top_themes': []}
            
        # Сбор топ слов
        all_words = []
        for text in cluster_data['processed_text']:
            all_words.extend(text.split())
        
        top_themes = Counter(all_words).most_common(10)
        
        return {
            'size': len(cluster_data),
            'top_themes': top_themes
        }

    def predict_cluster_for_text(self, text):
        """Предсказание для пользователя"""
        processed_text = self.preprocess_text(text)
        if not processed_text:
            return {'cluster': -1, 'confidence': 0.0, 'error': 'Пустой текст'}
            
        vec = self.model.encode([processed_text])
        cluster = self.kmeans.predict(vec)[0]
        dist = self.kmeans.transform(vec)[0][cluster]
        confidence = 1.0 / (1.0 + dist)
        
        cluster_info = self.get_cluster_info(cluster)
        
        return {
            'cluster': int(cluster),
            'confidence': float(confidence),
            'processed_text': processed_text,
            'cluster_size': cluster_info['size'],
            'top_themes': cluster_info['top_themes']
        }

    def find_similar_profiles(self, user_text, top_k=20):
        processed = self.preprocess_text(user_text)
        if not processed: return pd.DataFrame()
        
        vec = self.model.encode([processed])
        sims = cosine_similarity(vec, self.embeddings)[0]
        indices = np.argsort(sims)[::-1][:top_k]
        
        results = []
        for idx in indices:
            results.append({
                'index': self.df.index[idx], 
                'similarity': sims[idx],
                'description': self.df.iloc[idx]['Описание'],
                'cluster': self.df.iloc[idx]['cluster']
            })
        return pd.DataFrame(results)

    def get_dataset_stats(self):
        """Статистика для боковой панели приложения"""
        if self.df is None: return None
        
        stats = {
            'total_profiles': len(self.df),
            'clusters_count': len(self.df['cluster'].unique()) if 'cluster' in self.df.columns else 0,
        }
        if 'cluster' in self.df.columns:
            stats['cluster_sizes'] = self.df['cluster'].value_counts().to_dict()
        if self.embeddings is not None:
            stats['embedding_dimensions'] = self.embeddings.shape[1]
        if self.similarity_matrix is not None:
            triu_indices = np.triu_indices_from(self.similarity_matrix, k=1)
            stats['avg_similarity'] = np.mean(self.similarity_matrix[triu_indices])
            
        return stats

    def save_processed_data(self, output_path='processed_base_doc.xlsx'):
        if self.df is not None:
            self.df.to_excel(output_path, index=False)

    def load_and_process_data(self, excel_path='base_doc.xlsx', n_clusters=6):
        self.load_and_clean_data(excel_path)
        self.create_bert_embeddings()
        self.perform_clustering(n_clusters)
        self.save_processed_data()
        return self.df, self.embeddings

bert_processor = BERTProcessor()

def initialize_processor():
    """Инициализирует процессор и возвращает данные"""
    return bert_processor.load_and_process_data()

def predict_user_cluster(user_text):
    """Определяет кластер для текста пользователя"""
    return bert_processor.predict_cluster_for_text(user_text)

def find_similar_profiles(user_text, top_k=20):
    """Поиск похожих профилей через обертку"""
    return bert_processor.find_similar_profiles(user_text, top_k)