import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict
import time
from bert_processor import bert_processor, initialize_processor, predict_user_cluster

st.set_page_config(
    page_title="FriendFinder - AI Powered Friend Matching",
    page_icon="🤝",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 4rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(46, 125, 50, 0.3);
        font-weight: bold;
    }
    .hero-section {
        background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
        padding: 40px 20px;
        border-radius: 20px;
        margin: 15px 0;
        color: white;
        text-align: center;
        box-shadow: 0 8px 25px rgba(76, 175, 80, 0.3);
    }
    .profile-input-section {
        background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
        padding: 20px;
        border-radius: 15px;
        margin: 15px 0;
        border: 2px solid #388E3C;
        box-shadow: 0 6px 20px rgba(76, 175, 80, 0.2);
        color: white;
    }
    .profile-card {
        border: 2px solid #4CAF50;
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        background: linear-gradient(135deg, #66BB6A 0%, #43A047 100%);
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.15);
        color: white;
    }
    .match-score {
        font-size: 2rem;
        font-weight: bold;
        color: white;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .feedback-buttons {
        display: flex;
        justify-content: center;
        gap: 20px;
        margin-top: 20px;
    }
    .similarity-bar {
        height: 15px;
        background: linear-gradient(90deg, #81C784 0%, #4CAF50 100%);
        border-radius: 8px;
        margin: 10px 0;
        box-shadow: 0 2px 6px rgba(76, 175, 80, 0.3);
    }
    .stats-container {
        background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
        padding: 20px;
        border-radius: 15px;
        margin: 15px 0;
        border: 2px solid #388E3C;
        color: white;
    }
    .sidebar-section {
        background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
        border: 2px solid #388E3C;
        color: white;
    }
    .sidebar-header {
        color: white !important;
        font-weight: bold !important;
        font-size: 1.8rem !important;
        text-align: center;
        margin-bottom: 12px !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    .stButton button {
        background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 20px 35px;
        font-weight: bold;
        font-size: 2rem;
        box-shadow: 0 4px 12px rgba(76, 175, 80, 0.3);
        transition: all 0.3s ease;
        height: auto;
        min-height: 70px;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(76, 175, 80, 0.4);
        background: linear-gradient(135deg, #2E7D32 0%, #4CAF50 100%);
    }
    .main-content {
        font-size: 1.2rem;
        line-height: 1.5;
        color: white;
    }
    h1, h2, h3, h4 {
        color: white !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    h2 {
        font-size: 2rem !important;
        margin-bottom: 1.2rem !important;
    }
    h3 {
        font-size: 1.6rem !important;
        margin-bottom: 1rem !important;
    }
    .profile-description {
        font-size: 1.1rem !important;
        line-height: 1.5 !important;
        color: white !important;
        background: rgba(255,255,255,0.1);
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
        border-left: 4px solid #81C784;
        font-weight: 500;
        backdrop-filter: blur(10px);
    }
    .profile-description-title {
        font-size: 1.6rem !important;
        color: white !important;
        font-weight: bold !important;
        margin-bottom: 12px !important;
    }
    .stTextArea textarea {
        font-size: 1.2rem !important;
        line-height: 1.4 !important;
        padding: 12px !important;
        border-radius: 12px !important;
        border: 2px solid #388E3C !important;
        background: rgba(255,255,255,0.95) !important;
        color: #000000 !important;
    }
    .stTextArea textarea::placeholder {
        color: #666666 !important;
    }
    [data-testid="metric-container"] {
        padding: 25px !important;
        background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%) !important;
        border-radius: 15px !important;
        border: 2px solid #388E3C !important;
        color: white !important;
    }
    [data-testid="metric-value"] {
        font-size: 3.5rem !important;
        font-weight: bold !important;
        color: white !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    [data-testid="metric-label"] {
        font-size: 1.8rem !important;
        font-weight: bold !important;
        color: white !important;
    }
    .results-info {
        background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
        color: white !important;
        padding: 25px;
        border-radius: 20px;
        margin: 15px 0;
        text-align: center;
        border: 2px solid white;
        box-shadow: 0 6px 20px rgba(76, 175, 80, 0.3);
    }
    .success-message {
        background: linear-gradient(135deg, #66BB6A 0%, #4CAF50 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin: 15px 0;
        border: 2px solid white;
    }
    .truncated-title {
        font-size: 1rem !important;
        font-weight: 500 !important;
        color: white !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    .expander-content {
        background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
        color: white;
        padding: 15px;
        border-radius: 12px;
        margin: 8px 0;
    }
    /* Уменьшаем ширину основного контента */
    .main .block-container {
        max-width: 800px;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    /* Улучшенные стили для текстового поля */
    .stTextArea > div > div {
        background: white !important;
    }
    .stTextArea > div > div > textarea {
        color: #000000 !important;
    }
    /* Стили для placeholder */
    .stTextArea textarea::placeholder {
        color: #666666 !important;
        opacity: 1 !important;
    }
    /* Убираем лишние отступы */
    .css-1d391kg {
        padding: 0.5rem 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_processor():
    """Загружает BERT модель и обрабатывает данные профилей"""
    with st.spinner('🔄 Загружаем AI модель и обрабатываем данные... Это может занять несколько минут...'):
        df, embeddings = initialize_processor()
        return df, embeddings, bert_processor

def initialize_session_state():
    """Инициализирует состояние приложения при первом запуске"""
    defaults = {
        'current_profile_index': 0,
        'recommendations': None,
        'user_feedback': defaultdict(list),
        'user_profile': "",
        'search_performed': False,
        'processor_loaded': False,
        'user_cluster': None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def display_welcome_section():
    """Показывает главный заголовок и описание приложения"""
    st.markdown("""
    <div class="hero-section">
        <h1 style='font-size: 3.5rem; margin-bottom: 15px;'>🤝 FriendFinder AI</h1>
        <p style='font-size: 1.6rem; margin-bottom: 12px;'>Находите друзей по интересам с помощью искусственного интеллекта</p>
        <p style='font-size: 1.2rem; opacity: 0.9;'>BERT + Кластеризация + Семантический анализ</p>
        <div style='margin-top: 20px; font-size: 1.1rem;'>
            <span style='background: rgba(255,255,255,0.2); padding: 8px 16px; border-radius: 20px; margin: 0 8px;'>🚀 Быстро</span>
            <span style='background: rgba(255,255,255,0.2); padding: 8px 16px; border-radius: 20px; margin: 0 8px;'>🎯 Точно</span>
            <span style='background: rgba(255,255,255,0.2); padding: 8px 16px; border-radius: 20px; margin: 0 8px;'>🤖 Умно</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_profile_input_section():
    """Отображает текстовое поле для ввода описания пользователя"""
    st.markdown('<div class="profile-input-section">', unsafe_allow_html=True)
    
    st.markdown("### 💫 Расскажите о себе и своих интересах")
    
    user_profile = st.text_area(
        "Опишите ваши увлечения, хобби, интересы, чем хотели бы заниматься с друзьями:",
        height=150,
        value=st.session_state.user_profile,
        placeholder="💬 Например: Меня зовут Алексей, мне 28 лет. Увлекаюсь программированием, люблю активный отдых, походы в горы, настольные игры...",
        label_visibility="collapsed"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    return user_profile

def display_sidebar_stats(processor):
    """Показывает статистику и информацию в боковой панели"""
    stats = processor.get_dataset_stats()
    
    with st.sidebar:
        st.markdown(f"""
        <div class="sidebar-section" style='text-align: center;'>
            <div class="sidebar-header">🌱 AI POWERED</div>
            <p style='font-size: 1.1rem; font-weight: bold;'>BERT + Кластеризация</p>
            <div style='margin-top: 12px;'>
                <span style='background: rgba(255,255,255,0.2); padding: 4px 12px; border-radius: 12px;'>🤖 ML</span>
                <span style='background: rgba(255,255,255,0.2); padding: 4px 12px; border-radius: 12px;'>🔍 NLP</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("### 📊 Статистика системы")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("👥 Профилей", stats['total_profiles'])
        with col2:
            st.metric("🎯 Групп", stats['clusters_count'])
        
        col3, col4 = st.columns(2)
        with col3:
            st.metric("📐 Размерность", f"{stats['embedding_dimensions']}D")
        with col4:
            st.metric("💫 Схожесть", f"{stats['avg_similarity']:.3f}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("### 🎯 Группы по интересам")
        for cluster_id, size in stats['cluster_sizes'].items():
            with st.expander(f"Группа {cluster_id} ({size} участников)"):
                try:
                    cluster_info = processor.get_cluster_info(cluster_id)
                    st.write("**Топ-интересы:**")
                    for i, (word, count) in enumerate(cluster_info['top_themes'][:5], 1):
                        st.markdown(f"<div style='color: white;'>{i}. {word} ({count})</div>", unsafe_allow_html=True)
                except Exception as e:
                    st.write("Информация временно недоступна")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("### 💡 Советы для анкеты:")
        tips = [
            "🎨 Опишите ваши увлечения и хобби",
            "🎮 Укажите любимые занятия и игры", 
            "📚 Расскажите о сферах интересов",
            "🏃‍♂️ Опишите активный отдых, которым занимаетесь",
            "🎵 Укажите музыкальные/кино предпочтения",
            "😊 Будьте искренними и открытыми",
            "📝 Пишите развернуто, но лаконично"
        ]
        for tip in tips:
            st.markdown(f"<div style='color: white; margin: 6px 0;'>• {tip}</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

def display_search_results(processor, user_profile):
    """Ищет похожие профили используя AI модель"""
    with st.spinner('🔍 AI анализирует ваши интересы и ищет единомышленников...'):
        user_cluster = predict_user_cluster(user_profile)
        st.session_state.user_cluster = user_cluster
        
        recommendations = processor.find_similar_profiles(user_profile)
        st.session_state.recommendations = recommendations
        st.session_state.current_profile_index = 0
        st.session_state.search_performed = True
    
    return True

def display_current_profile(recommendations, current_index):
    """Показывает текущий профиль из рекомендаций"""
    if current_index >= len(recommendations):
        return False
        
    current_profile = recommendations.iloc[current_index]
    
    profile_col1, profile_col2 = st.columns([2, 1])
    
    with profile_col1:
        st.markdown(f'<div class="profile-card">', unsafe_allow_html=True)
        
        similarity_percent = current_profile['similarity'] * 100
        st.markdown(f'### 🤝 Совместимость по интересам: <span class="match-score">{similarity_percent:.1f}%</span>', unsafe_allow_html=True)
        
        st.markdown(f'<div class="similarity-bar" style="width: {similarity_percent}%"></div>', unsafe_allow_html=True)
        
        st.markdown(f'### 🎯 Группа интересов: <span style="color: white; font-weight: bold;">#{current_profile["cluster"]}</span>', unsafe_allow_html=True)
        
        if st.session_state.user_cluster:
            user_cluster_num = st.session_state.user_cluster['cluster']
            current_cluster_num = current_profile['cluster']
            
            if user_cluster_num == current_cluster_num:
                st.markdown(f'### 🎪 **ОДНА ГРУППА ИНТЕРЕСОВ!** Оба в группе #{user_cluster_num}')
            else:
                st.markdown(f'### 🔀 Разные группы: вы в #{user_cluster_num}, анкета в #{current_cluster_num}')
        
        st.markdown('<div class="profile-description-title">📖 ОПИСАНИЕ ИНТЕРЕСОВ:</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="profile-description">{current_profile["description"]}</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with profile_col2:
        st.markdown("### 📈 Детали совместимости")
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = similarity_percent,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Уровень совместимости", 'font': {'size': 18, 'color': 'white'}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': "#81C784", 'thickness': 0.25},
                'bgcolor': "rgba(255,255,255,0.1)",
                'borderwidth': 2,
                'bordercolor': "white",
                'steps': [
                    {'range': [0, 25], 'color': "rgba(255,255,255,0.1)"},
                    {'range': [25, 50], 'color': "rgba(129, 199, 132, 0.3)"},
                    {'range': [50, 75], 'color': "rgba(129, 199, 132, 0.6)"},
                    {'range': [75, 100], 'color': "rgba(129, 199, 132, 0.9)"}
                ]
            }
        ))
        
        fig.update_layout(
            height=300,
            margin=dict(l=15, r=15, t=60, b=15),
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color': "white", 'size': 16},
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        current_num = current_index + 1
        total_num = len(recommendations)
        progress = current_num / total_num
        
        st.markdown(f"""
        <div class="stats-container">
            <div style="text-align: center;">
                <div style="font-size: 1.3rem; margin-bottom: 8px;">📄 Текущая анкета</div>
                <div style="font-size: 1.8rem; font-weight: bold;">{current_num} из {total_num}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 12px; margin-top: 15px;">
            <div style="color: white; font-size: 1.1rem; margin-bottom: 8px;">Прогресс просмотра:</div>
            <div style="width: 100%; background: rgba(255,255,255,0.2); border-radius: 8px; overflow: hidden;">
                <div style="width: {progress*100}%; height: 15px; background: linear-gradient(90deg, #81C784 0%, #4CAF50 100%); transition: all 0.3s ease;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    return True

def display_feedback_buttons():
    """Показывает кнопки для оценки профилей"""
    st.markdown("---")
    st.markdown("### 💭 Интересен ли вам этот человек?")
    
    feedback_col1, feedback_col2, feedback_col3 = st.columns(3)
    
    with feedback_col1:
        if st.button("👍 ИНТЕРЕСЕН", key="like_btn", use_container_width=True):
            current_profile = st.session_state.recommendations.iloc[st.session_state.current_profile_index]
            st.session_state.user_feedback['liked'].append(current_profile['index'])
            st.session_state.current_profile_index += 1
            st.rerun()
    
    with feedback_col2:
        if st.button("👎 НЕ ИНТЕРЕСЕН", key="dislike_btn", use_container_width=True):
            current_profile = st.session_state.recommendations.iloc[st.session_state.current_profile_index]
            st.session_state.user_feedback['disliked'].append(current_profile['index'])
            st.session_state.current_profile_index += 1
            st.rerun()
    
    with feedback_col3:
        if st.button("⏭️ СЛЕДУЮЩИЙ", key="skip_btn", use_container_width=True):
            st.session_state.current_profile_index += 1
            st.rerun()

def display_search_stats(recommendations):
    """Показывает статистику текущего поиска"""
    st.markdown('<div class="stats-container">', unsafe_allow_html=True)
    
    stats_col1, stats_col2, stats_col3, stats_col4, stats_col5 = st.columns(5)
    
    with stats_col1:
        st.metric("📊 Всего анкет", len(recommendations))
    with stats_col2:
        st.metric("👍 Заинтересовали", len(st.session_state.user_feedback['liked']))
    with stats_col3:
        st.metric("👎 Не заинтересовали", len(st.session_state.user_feedback['disliked']))
    with stats_col4:
        if st.session_state.current_profile_index < len(recommendations):
            current_cluster = recommendations.iloc[st.session_state.current_profile_index]['cluster']
            st.metric("🎯 Группа анкеты", f"#{current_cluster}")
        else:
            st.metric("🎯 Группа анкеты", "N/A")
    with stats_col5:
        remaining = len(recommendations) - st.session_state.current_profile_index
        st.metric("⏳ Осталось", max(0, remaining))
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_results_header(recommendations):
    """Показывает заголовок раздела с результатами"""
    user_cluster = st.session_state.user_cluster
    
    st.markdown(f"""
    <div class="results-info">
        <h2 style='margin-bottom: 12px;'>🎉 Результаты поиска</h2>
        <p style='font-size: 1.4rem; margin-bottom: 8px;'>Найдено <strong>{len(recommendations)}</strong> человек со схожими интересами</p>
        <p style='font-size: 1.2rem; margin-bottom: 5px;'>🎯 Ваши интересы относятся к <strong>группе #{user_cluster['cluster']}</strong></p>
        <div style='margin-top: 15px;'>
            <span style='background: rgba(255,255,255,0.2); padding: 6px 12px; border-radius: 15px; margin: 0 4px;'>🤖 AI Powered</span>
            <span style='background: rgba(255,255,255,0.2); padding: 6px 12px; border-radius: 15px; margin: 0 4px;'>🎯 Точный подбор</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_final_results(recommendations, processor):
    """Показывает итоговые результаты после просмотра всех профилей"""
    st.markdown('<div class="success-message">', unsafe_allow_html=True)
    st.markdown("### 🎉 Вы просмотрели все рекомендованные анкеты!")
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("### 📊 Итоги вашего поиска")
    
    final_col1, final_col2, final_col3, final_col4 = st.columns(4)
    
    with final_col1:
        st.metric("👀 Всего просмотрено", len(recommendations))
    with final_col2:
        st.metric("🤝 Заинтересовали", len(st.session_state.user_feedback['liked']))
    with final_col3:
        st.metric("💔 Не заинтересовали", len(st.session_state.user_feedback['disliked']))
    with final_col4:
        success_rate = (len(st.session_state.user_feedback['liked']) / len(recommendations)) * 100 if len(recommendations) > 0 else 0
        st.metric("🎯 Успешность подбора", f"{success_rate:.1f}%")
    
    if st.button("🌱 НАЧАТЬ НОВЫЙ ПОИСК", key="new_search_btn", use_container_width=True):
        st.session_state.current_profile_index = 0
        st.session_state.recommendations = None
        st.session_state.search_performed = False
        st.session_state.user_feedback = defaultdict(list)
        st.session_state.user_cluster = None
        st.rerun()
    
    if st.session_state.user_feedback['liked']:
        st.markdown("### 💖 Вам понравились эти люди:")
        
        for profile_idx in st.session_state.user_feedback['liked']:
            try:
                profile = processor.df.iloc[profile_idx]
                description_text = profile["Описание"]
                if len(description_text) > 120:
                    title = description_text[:120] + "..."
                else:
                    title = description_text
                
                if len(title.strip()) == 0:
                    title = "Анкета без описания"
                
                with st.expander(f"💫 {title}"):
                    st.markdown(f'<div class="profile-description">{profile["Описание"]}</div>', unsafe_allow_html=True)
            except Exception as e:
                continue

def main():
    """Главная функция приложения"""
    initialize_session_state()
    display_welcome_section()
    
    try:
        df, embeddings, processor = load_processor()
        st.session_state.processor_loaded = True
    except Exception as e:
        st.error(f"❌ Ошибка загрузки процессора: {str(e)}")
        st.info("⚠️ Пожалуйста, убедитесь что файл base_doc.xlsx находится в корневой панели")
        return
    
    display_sidebar_stats(processor)
    
    user_profile = display_profile_input_section()
    
    if st.button("🌱 Найти единомышленников", use_container_width=True):
        if user_profile.strip():
            st.session_state.user_profile = user_profile
            if display_search_results(processor, user_profile):
                st.rerun()
        else:
            st.error("❌ Пожалуйста, заполните информацию о ваших интересах!")
    
    if (st.session_state.search_performed and 
        st.session_state.recommendations is not None and 
        len(st.session_state.recommendations) > 0):
        
        recommendations = st.session_state.recommendations
        
        display_results_header(recommendations)
        display_search_stats(recommendations)
        
        if st.session_state.current_profile_index < len(recommendations):
            if display_current_profile(recommendations, st.session_state.current_profile_index):
                display_feedback_buttons()
        else:
            display_final_results(recommendations, processor)
    
    elif st.session_state.search_performed:
        st.info("🔍 По вашему запросу не найдено подходящих людей. Попробуйте изменить описание ваших интересов.")

if __name__ == "__main__":
    main()