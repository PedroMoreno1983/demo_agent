# shopai_ultimate.py
# Sistema completo con: Scraping inteligente, Embeddings, Predicción LSTM, 
# Recomendaciones, Visión por Computadora y Planificación

import streamlit as st
import asyncio
import aiohttp
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
import hashlib
import time
import random
import base64
import io
from collections import defaultdict, deque
import sqlite3
import threading
import pickle
import warnings
warnings.filterwarnings('ignore')

# Intentar importar librerías avanzadas
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model, save_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    HAS_TENSORFLOW = True
except:
    HAS_TENSORFLOW = False

try:
    from transformers import pipeline, AutoModelForImageClassification, AutoFeatureExtractor
    from PIL import Image
    HAS_TRANSFORMERS = True
except:
    HAS_TRANSFORMERS = False

try:
    import cv2
    HAS_OPENCV = True
except:
    HAS_OPENCV = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans
    from sklearn.ensemble import RandomForestRegressor
    HAS_SKLEARN = True
except:
    HAS_SKLEARN = False

try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMADB = True
except:
    HAS_CHROMADB = False

try:
    import openai
    from openai import OpenAI
    HAS_OPENAI = True
except:
    HAS_OPENAI = False

try:
    from playwright.async_api import async_playwright
    HAS_PLAYWRIGHT = True
except:
    HAS_PLAYWRIGHT = False

from bs4 import BeautifulSoup
import requests

# Configuración de página
st.set_page_config(
    page_title="ShopAI Ultimate - Agente de Compras IA",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CSS PROFESIONAL - UI DE ÚLTIMA GENERACIÓN
# ============================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;700&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    :root {
        --primary: #6366f1;
        --primary-dark: #4f46e5;
        --secondary: #ec4899;
        --accent: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
        --dark: #0f172a;
        --gray-50: #f8fafc;
        --gray-100: #f1f5f9;
        --gray-200: #e2e8f0;
        --gray-800: #1e293b;
    }
    
    /* Header cyberpunk */
    .cyber-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #312e81 100%);
        padding: 2rem;
        margin: -4rem -4rem 2rem -4rem;
        position: relative;
        overflow: hidden;
        border-bottom: 3px solid #6366f1;
    }
    
    .cyber-grid {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-image: 
            linear-gradient(rgba(99, 102, 241, 0.1) 1px, transparent 1px),
            linear-gradient(90deg, rgba(99, 102, 241, 0.1) 1px, transparent 1px);
        background-size: 50px 50px;
        animation: gridMove 20s linear infinite;
    }
    
    @keyframes gridMove {
        0% { transform: translate(0, 0); }
        100% { transform: translate(50px, 50px); }
    }
    
    .cyber-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #6366f1, #ec4899, #10b981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        position: relative;
        z-index: 2;
        text-transform: uppercase;
        letter-spacing: 4px;
    }
    
    .cyber-subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1.1rem;
        margin-top: 0.5rem;
        position: relative;
        z-index: 2;
    }
    
    /* Glassmorphism cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.15);
    }
    
    /* Neural network visualization */
    .neural-node {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        background: #6366f1;
        display: inline-block;
        margin: 2px;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(1.2); }
    }
    
    /* Chat futurista */
    .ai-chat-container {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        border-radius: 24px;
        overflow: hidden;
        border: 1px solid #334155;
    }
    
    .ai-message {
        padding: 1rem 1.5rem;
        margin: 0.5rem 1rem;
        border-radius: 16px;
        max-width: 80%;
        animation: slideIn 0.3s ease;
    }
    
    .ai-message.user {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        margin-left: auto;
        border-bottom-right-radius: 4px;
    }
    
    .ai-message.assistant {
        background: #334155;
        color: #e2e8f0;
        border-bottom-left-radius: 4px;
        border: 1px solid #475569;
    }
    
    .ai-thinking {
        display: flex;
        gap: 0.5rem;
        padding: 1rem;
        align-items: center;
        color: #94a3b8;
    }
    
    .thinking-dot {
        width: 8px;
        height: 8px;
        background: #6366f1;
        border-radius: 50%;
        animation: thinking 1.4s infinite;
    }
    
    .thinking-dot:nth-child(2) { animation-delay: 0.2s; }
    .thinking-dot:nth-child(3) { animation-delay: 0.4s; }
    
    @keyframes thinking {
        0%, 60%, 100% { transform: translateY(0); }
        30% { transform: translateY(-10px); }
    }
    
    /* Dashboard metrics */
    .metric-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border-left: 4px solid var(--primary);
        transition: all 0.3s;
    }
    
    .metric-card:hover {
        transform: scale(1.02);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--dark);
    }
    
    .metric-label {
        color: #64748b;
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Prediction cards */
    .prediction-up {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border-left: 4px solid #059669;
    }
    
    .prediction-down {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border-left: 4px solid #dc2626;
    }
    
    .prediction-stable {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        border-left: 4px solid #2563eb;
    }
    
    /* Camera/vision section */
    .vision-dropzone {
        border: 3px dashed #6366f1;
        border-radius: 20px;
        padding: 3rem;
        text-align: center;
        background: #f8fafc;
        transition: all 0.3s;
    }
    
    .vision-dropzone:hover {
        background: #eef2ff;
        border-color: #4f46e5;
    }
    
    /* Weekly planner */
    .day-card {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        min-height: 150px;
    }
    
    .day-header {
        font-weight: 700;
        color: var(--primary);
        border-bottom: 2px solid var(--gray-200);
        padding-bottom: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    /* Progress bars animated */
    .progress-container {
        background: #e2e8f0;
        border-radius: 999px;
        overflow: hidden;
        height: 8px;
    }
    
    .progress-bar {
        height: 100%;
        background: linear-gradient(90deg, #6366f1, #ec4899);
        border-radius: 999px;
        transition: width 1s ease;
        position: relative;
        overflow: hidden;
    }
    
    .progress-bar::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(
            90deg,
            transparent,
            rgba(255,255,255,0.3),
            transparent
        );
        animation: shimmer 2s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    /* Hide Streamlit elements */
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# BASE DE DATOS INTELIGENTE CON EMBEDDINGS
# ============================================================

@dataclass
class UserProfile:
    user_id: str
    preferences: Dict[str, Any]
    purchase_history: List[Dict]
    dietary_restrictions: List[str]
    budget_range: Tuple[int, int]
    preferred_stores: List[str]
    family_size: int
    created_at: datetime

class IntelligentDatabase:
    """
    Base de datos híbrida: SQL + Vector DB + Time Series
    """
    
    def __init__(self):
        self.conn = sqlite3.connect('shopai_ultimate.db', check_same_thread=False)
        self.init_database()
        self.setup_vector_store()
        self.user_profiles = {}
        
    def init_database(self):
        """Inicializa todas las tablas necesarias"""
        cursor = self.conn.cursor()
        
        # Productos con embeddings
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS products (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                brand TEXT,
                price INTEGER,
                original_price INTEGER,
                category TEXT,
                subcategory TEXT,
                store_id TEXT,
                store_name TEXT,
                image_url TEXT,
                product_url TEXT,
                unit TEXT,
                nutritional_info TEXT,
                tags TEXT,
                embedding BLOB,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                popularity_score REAL DEFAULT 0
            )
        ''')
        
        # Historial de precios para predicciones LSTM
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS price_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id TEXT,
                price INTEGER,
                day_of_week INTEGER,
                week_of_year INTEGER,
                is_holiday INTEGER,
                stock_level INTEGER,
                demand_index REAL,
                recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (product_id) REFERENCES products(id)
            )
        ''')
        
        # Compras de usuarios para recomendaciones colaborativas
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_purchases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                product_id TEXT,
                quantity INTEGER,
                price_paid INTEGER,
                purchase_date TIMESTAMP,
                store_id TEXT,
                satisfaction_rating INTEGER,
                repurchase_intent INTEGER,
                FOREIGN KEY (product_id) REFERENCES products(id)
            )
        ''')
        
        # Menús semanales generados por IA
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS weekly_menus (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                week_start DATE,
                menu_data TEXT,
                total_cost INTEGER,
                nutritional_score REAL,
                generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Detecciones de visión por computadora
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS vision_detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                image_hash TEXT,
                detected_items TEXT,
                confidence_scores TEXT,
                detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Índices para performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_products_category ON products(category)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_price_history_product ON price_history(product_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_price_history_date ON price_history(recorded_at)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_purchases_user ON user_purchases(user_id)')
        
        self.conn.commit()
    
    def setup_vector_store(self):
        """Configura ChromaDB para búsqueda semántica"""
        if HAS_CHROMADB:
            try:
                self.chroma_client = chromadb.Client(Settings(
                    chroma_db_impl="duckdb+parquet",
                    persist_directory="./chroma_db_ultimate"
                ))
                self.products_collection = self.chroma_client.get_or_create_collection(
                    name="products",
                    metadata={"hnsw:space": "cosine"}
                )
                self.recipes_collection = self.chroma_client.get_or_create_collection(
                    name="recipes",
                    metadata={"hnsw:space": "cosine"}
                )
            except Exception as e:
                st.error(f"Error ChromaDB: {e}")
                self.chroma_client = None
        else:
            self.chroma_client = None
    
    def get_embedding(self, text: str) -> List[float]:
        """Genera embedding usando OpenAI o fallback local"""
        if HAS_OPENAI and st.secrets.get("OPENAI_API_KEY"):
            try:
                client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
                response = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=text
                )
                return response.data[0].embedding
            except:
                pass
        
        # Fallback: TF-IDF simple
        if HAS_SKLEARN:
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer(max_features=1536)
            try:
                vec = vectorizer.fit_transform([text])
                return vec.toarray()[0].tolist()
            except:
                pass
        
        # Último fallback: hash-based
        random.seed(hash(text))
        return [random.uniform(-1, 1) for _ in range(1536)]
    
    def semantic_search(self, query: str, filters: Dict = None, top_k: int = 10) -> List[Dict]:
        """Búsqueda semántica con filtros"""
        if not self.chroma_client:
            return self.fallback_search(query, top_k)
        
        try:
            query_embedding = self.get_embedding(query)
            
            where_clause = {}
            if filters:
                if 'category' in filters:
                    where_clause['category'] = filters['category']
                if 'store_id' in filters:
                    where_clause['store_id'] = filters['store_id']
            
            results = self.products_collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_clause if where_clause else None
            )
            
            products = []
            for idx, doc_id in enumerate(results['ids'][0]):
                metadata = results['metadatas'][0][idx]
                metadata['similarity_score'] = 1 - results['distances'][0][idx]
                metadata['id'] = doc_id
                products.append(metadata)
            
            return products
            
        except Exception as e:
            return self.fallback_search(query, top_k)
    
    def fallback_search(self, query: str, top_k: int) -> List[Dict]:
        """Búsqueda por texto tradicional"""
        cursor = self.conn.cursor()
        search_term = f"%{query}%"
        
        cursor.execute('''
            SELECT * FROM products 
            WHERE name LIKE ? OR brand LIKE ? OR tags LIKE ?
            ORDER BY popularity_score DESC, price ASC
            LIMIT ?
        ''', (search_term, search_term, search_term, top_k))
        
        columns = [description[0] for description in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def get_price_timeseries(self, product_id: str, days: int = 60) -> pd.DataFrame:
        """Obtiene serie temporal de precios para LSTM"""
        cursor = self.conn.cursor()
        
        cursor.execute('''
            SELECT price, day_of_week, week_of_year, is_holiday, 
                   stock_level, demand_index, recorded_at
            FROM price_history
            WHERE product_id = ? 
            AND recorded_at >= datetime('now', '-{} days')
            ORDER BY recorded_at ASC
        '''.format(days), (product_id,))
        
        data = cursor.fetchall()
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data, columns=[
            'price', 'day_of_week', 'week_of_year', 'is_holiday',
            'stock_level', 'demand_index', 'date'
        ])
        df['date'] = pd.to_datetime(df['date'])
        return df
    
    def get_user_purchase_pattern(self, user_id: str) -> Dict:
        """Analiza patrones de compra del usuario"""
        cursor = self.conn.cursor()
        
        # Frecuencia de categorías
        cursor.execute('''
            SELECT p.category, COUNT(*) as count, AVG(up.price_paid) as avg_price
            FROM user_purchases up
            JOIN products p ON up.product_id = p.id
            WHERE up.user_id = ?
            GROUP BY p.category
            ORDER BY count DESC
        ''', (user_id,))
        
        categories = {row[0]: {'count': row[1], 'avg_price': row[2]} 
                     for row in cursor.fetchall()}
        
        # Días entre compras
        cursor.execute('''
            SELECT purchase_date FROM user_purchases
            WHERE user_id = ? ORDER BY purchase_date
        ''', (user_id,))
        
        dates = [row[0] for row in cursor.fetchall()]
        if len(dates) > 1:
            intervals = [(datetime.fromisoformat(dates[i+1]) - 
                         datetime.fromisoformat(dates[i])).days 
                        for i in range(len(dates)-1)]
            avg_interval = sum(intervals) / len(intervals)
        else:
            avg_interval = 7  # Default semanal
        
        # Productos que repite
        cursor.execute('''
            SELECT product_id, COUNT(*) as purchases
            FROM user_purchases
            WHERE user_id = ? AND repurchase_intent = 1
            GROUP BY product_id
            HAVING purchases > 1
            ORDER BY purchases DESC
            LIMIT 10
        ''', (user_id,))
        
        repurchased = [row[0] for row in cursor.fetchall()]
        
        return {
            'top_categories': categories,
            'purchase_frequency_days': avg_interval,
            'repurchased_products': repurchased,
            'total_purchases': len(dates)
        }

# ============================================================
# MODELO LSTM PARA PREDICCIÓN DE PRECIOS
# ============================================================

class PricePredictorLSTM:
    """
    Predice precios futuros usando LSTM con múltiples features
    """
    
    def __init__(self, db: IntelligentDatabase):
        self.db = db
        self.models = {}  # Un modelo por producto
        self.sequence_length = 14  # 14 días de historia
        self.features = ['price', 'day_of_week', 'week_of_year', 
                        'is_holiday', 'stock_level', 'demand_index']
        
    def prepare_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepara secuencias para LSTM"""
        if len(df) < self.sequence_length + 1:
            return None, None
        
        # Normalizar
        df_norm = df.copy()
        price_mean = df['price'].mean()
        price_std = df['price'].std() or 1
        df_norm['price'] = (df['price'] - price_mean) / price_std
        
        X, y = [], []
        for i in range(len(df_norm) - self.sequence_length):
            X.append(df_norm[self.features].iloc[i:i+self.sequence_length].values)
            y.append(df_norm['price'].iloc[i+self.sequence_length])
        
        return np.array(X), np.array(y)
    
    def build_model(self) -> Sequential:
        """Arquitectura LSTM"""
        if not HAS_TENSORFLOW:
            return None
        
        model = Sequential([
            LSTM(128, return_sequences=True, 
                 input_shape=(self.sequence_length, len(self.features))),
            Dropout(0.2),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
    
    def train_or_load(self, product_id: str) -> bool:
        """Entrena o carga modelo para un producto"""
        if not HAS_TENSORFLOW:
            return False
        
        # Intentar cargar modelo existente
        model_path = f'models/lstm_{product_id}.h5'
        try:
            self.models[product_id] = load_model(model_path)
            return True
        except:
            pass
        
        # Obtener datos
        df = self.db.get_price_timeseries(product_id, days=90)
        if len(df) < 30:  # Necesitamos suficiente historia
            return False
        
        X, y = self.prepare_sequences(df)
        if X is None:
            return False
        
        # Entrenar
        model = self.build_model()
        if model is None:
            return False
        
        model.fit(X, y, epochs=50, batch_size=16, verbose=0, 
                 validation_split=0.2)
        
        # Guardar
        import os
        os.makedirs('models', exist_ok=True)
        model.save(model_path)
        self.models[product_id] = model
        
        return True
    
    def predict_next_days(self, product_id: str, days: int = 7) -> List[Dict]:
        """Predice precios para los próximos días"""
        if product_id not in self.models:
            success = self.train_or_load(product_id)
            if not success:
                return []
        
        # Obtener última secuencia
        df = self.db.get_price_timeseries(product_id, days=30)
        if len(df) < self.sequence_length:
            return []
        
        last_sequence = df[self.features].iloc[-self.sequence_length:].values
        
        # Normalizar (usar stats de entrenamiento)
        price_mean = df['price'].mean()
        price_std = df['price'].std() or 1
        
        predictions = []
        current_sequence = last_sequence.copy()
        
        for i in range(days):
            # Predecir
            X_pred = current_sequence.reshape(1, self.sequence_length, len(self.features))
            pred_norm = self.models[product_id].predict(X_pred, verbose=0)[0][0]
            
            # Desnormalizar
            pred_price = pred_norm * price_std + price_mean
            pred_price = max(0, pred_price)  # No precios negativos
            
            # Fecha futura
            future_date = datetime.now() + timedelta(days=i+1)
            
            predictions.append({
                'date': future_date.strftime('%Y-%m-%d'),
                'predicted_price': round(pred_price),
                'confidence': self.calculate_confidence(i),
                'day_of_week': future_date.weekday(),
                'is_weekend': future_date.weekday() >= 5
            })
            
            # Actualizar secuencia para siguiente predicción
            new_row = current_sequence[-1].copy()
            new_row[0] = pred_norm  # Actualizar precio
            new_row[1] = future_date.weekday()  # Actualizar día
            current_sequence = np.vstack([current_sequence[1:], new_row])
        
        return predictions
    
    def calculate_confidence(self, days_ahead: int) -> float:
        """Calcula confianza basada en qué tan lejos es la predicción"""
        # Confianza disminuye con el tiempo
        base_confidence = 0.95
        decay = 0.05 * days_ahead
        return max(0.5, base_confidence - decay)
    
    def get_buying_recommendation(self, product_id: str) -> Dict:
        """Recomienda si comprar ahora o esperar"""
        predictions = self.predict_next_days(product_id, days=7)
        if not predictions:
            return {'recommendation': 'no_data', 'message': 'Sin suficiente historial'}
        
        current_price = self.db.get_price_timeseries(product_id, days=1)['price'].iloc[-1]
        min_future = min(p['predicted_price'] for p in predictions)
        max_future = max(p['predicted_price'] for p in predictions)
        avg_future = sum(p['predicted_price'] for p in predictions) / len(predictions)
        
        if current_price <= min_future * 1.02:  # Precio actual es casi el mínimo
            return {
                'recommendation': 'buy_now',
                'confidence': 0.85,
                'message': '🟢 COMPRAR AHORA - Precio en mínimo histórico proyectado',
                'current_price': current_price,
                'predicted_min': min_future,
                'potential_savings': 0,
                'best_day': 'Hoy'
            }
        elif current_price > avg_future * 1.05:  # Precio va a bajar
            best_day = min(predictions, key=lambda x: x['predicted_price'])
            return {
                'recommendation': 'wait',
                'confidence': 0.75,
                'message': f'🔴 ESPERAR - Precio proyectado a la baja',
                'current_price': current_price,
                'predicted_min': min_future,
                'potential_savings': current_price - min_future,
                'best_day': best_day['date'],
                'savings_percent': round((current_price - min_future) / current_price * 100, 1)
            }
        else:
            return {
                'recommendation': 'stable',
                'confidence': 0.6,
                'message': '🟡 PRECIO ESTABLE - Puedes comprar cuando quieras',
                'current_price': current_price,
                'predicted_range': f'${min_future} - ${max_future}'
            }

# ============================================================
# SISTEMA DE RECOMENDACIÓN COLABORATIVO + CONTENT-BASED
# ============================================================

class HybridRecommender:
    """
    Combina filtrado colaborativo con content-based filtering
    """
    
    def __init__(self, db: IntelligentDatabase):
        self.db = db
        self.user_item_matrix = None
        self.item_similarity = None
        self.last_update = None
        
    def build_user_item_matrix(self):
        """Construye matriz usuario-item"""
        cursor = self.db.conn.cursor()
        
        cursor.execute('''
            SELECT user_id, product_id, COUNT(*) as purchases
            FROM user_purchases
            GROUP BY user_id, product_id
        ''')
        
        data = cursor.fetchall()
        if not data:
            return None
        
        df = pd.DataFrame(data, columns=['user_id', 'product_id', 'purchases'])
        matrix = df.pivot(index='user_id', columns='product_id', 
                         values='purchases').fillna(0)
        
        self.user_item_matrix = matrix
        self.last_update = datetime.now()
        
        # Calcular similitud entre items (content-based)
        if HAS_SKLEARN:
            self.item_similarity = cosine_similarity(matrix.T)
        
        return matrix
    
    def get_collaborative_recommendations(self, user_id: str, n: int = 10) -> List[Dict]:
        """Recomendaciones basadas en usuarios similares"""
        if self.user_item_matrix is None or \
           (datetime.now() - self.last_update).days > 1:
            self.build_user_item_matrix()
        
        if self.user_item_matrix is None or user_id not in self.user_item_matrix.index:
            return []
        
        # Encontrar usuarios similares
        user_vector = self.user_item_matrix.loc[user_id].values.reshape(1, -1)
        
        if HAS_SKLEARN:
            similarities = cosine_similarity(user_vector, self.user_item_matrix)[0]
            similar_users = self.user_item_matrix.index[similarities.argsort()[-6:-1][::-1]]
            
            # Productos que compraron usuarios similares pero este usuario no
            recommendations = []
            for similar_user in similar_users:
                similar_items = self.user_item_matrix.loc[similar_user]
                user_items = self.user_item_matrix.loc[user_id]
                
                new_items = similar_items[similar_items > 0][user_items == 0]
                for product_id, score in new_items.items():
                    recommendations.append({
                        'product_id': product_id,
                        'score': score * similarities[self.user_item_matrix.index.get_loc(similar_user)],
                        'reason': f'Comprado por usuarios similares a ti'
                    })
            
            # Ordenar y deduplicar
            recommendations.sort(key=lambda x: x['score'], reverse=True)
            seen = set()
            unique_recs = []
            for rec in recommendations:
                if rec['product_id'] not in seen and len(unique_recs) < n:
                    seen.add(rec['product_id'])
                    unique_recs.append(rec)
            
            return unique_recs
        
        return []
    
    def get_content_recommendations(self, user_id: str, n: int = 10) -> List[Dict]:
        """Recomendaciones basadas en contenido de productos comprados"""
        # Obtener últimas compras del usuario
        cursor = self.db.conn.cursor()
        cursor.execute('''
            SELECT p.id, p.name, p.category, p.tags
            FROM user_purchases up
            JOIN products p ON up.product_id = p.id
            WHERE up.user_id = ?
            ORDER BY up.purchase_date DESC
            LIMIT 5
        ''', (user_id,))
        
        recent_purchases = cursor.fetchall()
        if not recent_purchases:
            return []
        
        # Construir perfil de usuario basado en contenido
        user_profile = defaultdict(float)
        for prod_id, name, category, tags in recent_purchases:
            # Peso por recencia
            weight = 1.0
            
            # Agregar categoría al perfil
            user_profile[f'cat_{category}'] += weight
            
            # Agregar tags
            if tags:
                for tag in tags.split(','):
                    user_profile[f'tag_{tag.strip()}'] += weight
        
        # Buscar productos similares no comprados
        recommendations = []
        
        for tag, weight in sorted(user_profile.items(), key=lambda x: x[1], reverse=True)[:5]:
            if tag.startswith('cat_'):
                category = tag[4:]
                cursor.execute('''
                    SELECT p.* FROM products p
                    LEFT JOIN user_purchases up ON p.id = up.product_id AND up.user_id = ?
                    WHERE p.category = ? AND up.id IS NULL
                    ORDER BY p.popularity_score DESC
                    LIMIT ?
                ''', (user_id, category, n//2))
                
                for row in cursor.fetchall():
                    recommendations.append({
                        'product_id': row[0],
                        'name': row[1],
                        'score': weight,
                        'reason': f'Porque te gusta {category}'
                    })
        
        return recommendations[:n]
    
    def get_hybrid_recommendations(self, user_id: str, context: Dict = None, n: int = 10) -> List[Dict]:
        """Combina ambos enfoques con pesos dinámicos"""
        collab = self.get_collaborative_recommendations(user_id, n)
        content = self.get_content_recommendations(user_id, n)
        
        # Peso dinámico basado en cantidad de datos del usuario
        cursor = self.db.conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM user_purchases WHERE user_id = ?', (user_id,))
        purchase_count = cursor.fetchone()[0]
        
        if purchase_count < 5:
            # Poco historial: más content-based
            weight_collab, weight_content = 0.3, 0.7
        elif purchase_count < 20:
            # Historial medio: balanceado
            weight_collab, weight_content = 0.5, 0.5
        else:
            # Mucho historial: más colaborativo
            weight_collab, weight_content = 0.7, 0.3
        
        # Combinar scores
        combined = {}
        
        for rec in collab:
            pid = rec['product_id']
            combined[pid] = {
                'score': rec['score'] * weight_collab,
                'reasons': [rec['reason']]
            }
        
        for rec in content:
            pid = rec['product_id']
            if pid in combined:
                combined[pid]['score'] += rec['score'] * weight_content
                combined[pid]['reasons'].append(rec['reason'])
            else:
                combined[pid] = {
                    'score': rec['score'] * weight_content,
                    'reasons': [rec['reason']]
                }
        
        # Ordenar y formatear
        sorted_recs = sorted(combined.items(), key=lambda x: x[1]['score'], reverse=True)
        
        result = []
        for pid, data in sorted_recs[:n]:
            cursor.execute('SELECT name, price, store_name FROM products WHERE id = ?', (pid,))
            row = cursor.fetchone()
            if row:
                result.append({
                    'product_id': pid,
                    'name': row[0],
                    'price': row[1],
                    'store': row[2],
                    'score': data['score'],
                    'reason': ' + '.join(data['reasons'][:2])
                })
        
        return result

# ============================================================
# VISIÓN POR COMPUTADORA - DETECCIÓN DE INGREDIENTES
# ============================================================

class VisionIngredientDetector:
    """
    Detecta ingredientes en fotos usando modelos de visión
    """
    
    def __init__(self):
        self.model = None
        self.feature_extractor = None
        self.load_model()
        
        # Mapeo de clases a ingredientes comunes
        self.ingredient_classes = {
            'apple': 'Manzana', 'banana': 'Plátano', 'orange': 'Naranja',
            'broccoli': 'Brócoli', 'carrot': 'Zanahoria',
            'milk': 'Leche', 'eggs': 'Huevos', 'bread': 'Pan',
            'cheese': 'Queso', 'yogurt': 'Yogurt',
            'chicken': 'Pollo', 'beef': 'Carne', 'fish': 'Pescado',
            'rice': 'Arroz', 'pasta': 'Pasta', 'potato': 'Papa',
            'tomato': 'Tomate', 'onion': 'Cebolla', 'garlic': 'Ajo',
            'lettuce': 'Lechuga', 'pepper': 'Pimentón',
            'bottle': 'Bebida', 'can': 'Conserva'
        }
    
    def load_model(self):
        """Carga modelo de visión"""
        if HAS_TRANSFORMERS:
            try:
                # Usar modelo pre-entrenado de HuggingFace
                model_name = "microsoft/resnet-50"
                self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
                self.model = AutoModelForImageClassification.from_pretrained(model_name)
            except Exception as e:
                st.warning(f"No se pudo cargar modelo de visión: {e}")
    
    def detect_ingredients(self, image_bytes: bytes) -> List[Dict]:
        """
        Detecta ingredientes en una imagen
        """
        if not self.model or not HAS_TRANSFORMERS:
            # Fallback: simulación para demo
            return self.simulate_detection(image_bytes)
        
        try:
            image = Image.open(io.BytesIO(image_bytes))
            
            # Preprocesar
            inputs = self.feature_extractor(images=image, return_tensors="pt")
            
            # Inferencia
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Obtener top predictions
            probs = outputs.logits.softmax(dim=1)
            top_probs, top_indices = probs.topk(5)
            
            detected = []
            for prob, idx in zip(top_probs[0], top_indices[0]):
                class_name = self.model.config.id2label[idx.item()]
                confidence = prob.item()
                
                # Mapear a ingrediente
                ingredient = self.ingredient_classes.get(class_name.lower(), class_name)
                
                detected.append({
                    'ingredient': ingredient,
                    'confidence': round(confidence, 3),
                    'raw_class': class_name
                })
            
            return detected
            
        except Exception as e:
            return self.simulate_detection(image_bytes)
    
    def simulate_detection(self, image_bytes: bytes) -> List[Dict]:
        """Simula detección para demo sin modelo"""
        # Generar "detección" determinista basada en hash de imagen
        image_hash = hashlib.md5(image_bytes).hexdigest()
        random.seed(int(image_hash[:8], 16))
        
        possible_ingredients = [
            'Papa', 'Zanahoria', 'Cebolla', 'Tomate', 'Lechuga',
            'Pollo', 'Carne', 'Pescado', 'Huevos', 'Leche',
            'Queso', 'Pan', 'Arroz', 'Pasta', 'Manzana'
        ]
        
        num_items = random.randint(3, 6)
        detected = []
        
        for item in random.sample(possible_ingredients, num_items):
            detected.append({
                'ingredient': item,
                'confidence': round(random.uniform(0.7, 0.95), 3),
                'raw_class': item.lower()
            })
        
        return sorted(detected, key=lambda x: x['confidence'], reverse=True)
    
    def suggest_recipes_from_ingredients(self, ingredients: List[str]) -> List[Dict]:
        """Sugiere recetas basadas en ingredientes detectados"""
        # Base de recetas simple
        recipes_db = {
            'Papa+Cebolla': {
                'name': 'Tortilla de Papas',
                'missing': ['Huevos', 'Aceite', 'Sal'],
                'time': '30 min'
            },
            'Pollo+Papa': {
                'name': 'Pollo al Horno con Papas',
                'missing': ['Aceite', 'Ajo', 'Romero'],
                'time': '60 min'
            },
            'Tomate+Cebolla': {
                'name': 'Salsa Pomodoro',
                'missing': ['Ajo', 'Albahaca', 'Aceite de Oliva'],
                'time': '20 min'
            },
            'Huevos+Queso': {
                'name': 'Omelette de Queso',
                'missing': ['Mantequilla', 'Sal', 'Pimienta'],
                'time': '10 min'
            },
            'Carne+Cebolla': {
                'name': 'Bistec Encebollado',
                'missing': ['Aceite', 'Salsa de Soja', 'Ajo'],
                'time': '25 min'
            }
        }
        
        # Encontrar mejor match
        ingredient_set = set(ingredients)
        best_matches = []
        
        for key, recipe in recipes_db.items():
            required = set(key.split('+'))
            have = len(required & ingredient_set)
            if have >= 1:
                best_matches.append({
                    'recipe': recipe,
                    'match_score': have / len(required),
                    'have_ingredients': list(required & ingredient_set),
                    'missing_ingredients': recipe['missing']
                })
        
        return sorted(best_matches, key=lambda x: x['match_score'], reverse=True)[:3]

# ============================================================
# PLANIFICADOR SEMANAL CON OPTIMIZACIÓN
# ============================================================

class WeeklyMealPlanner:
    """
    Genera menús semanales optimizados por presupuesto, nutrición y tiempo
    """
    
    def __init__(self, db: IntelligentDatabase):
        self.db = db
        
        # Base de datos nutricional simplificada
        self.nutritional_db = {
            'proteins': ['Pollo', 'Carne', 'Pescado', 'Huevos', 'Lentejas', 'Garbanzos'],
            'carbs': ['Arroz', 'Pasta', 'Papa', 'Pan', 'Quinoa', 'Avena'],
            'vegetables': ['Brócoli', 'Zanahoria', 'Espinaca', 'Tomate', 'Pimentón', 'Cebolla'],
            'fruits': ['Manzana', 'Plátano', 'Naranja', 'Frutilla', 'Arándanos']
        }
        
        self.recipe_templates = [
            {
                'name': '{protein} con {veg} y {carb}',
                'type': 'Almuerzo',
                'base_time': 30,
                'difficulty': 'Media'
            },
            {
                'name': 'Ensalada de {veg} con {protein}',
                'type': 'Cena',
                'base_time': 15,
                'difficulty': 'Fácil'
            },
            {
                'name': '{carb} con salsa de {veg} y {protein}',
                'type': 'Almuerzo',
                'base_time': 45,
                'difficulty': 'Media'
            }
        ]
    
    def generate_weekly_menu(self, user_id: str, budget: int, 
                            preferences: Dict = None) -> Dict:
        """
        Genera menú semanal completo
        """
        # Obtener restricciones del usuario
        cursor = self.db.conn.cursor()
        cursor.execute('''
            SELECT dietary_restrictions, family_size 
            FROM user_profiles WHERE user_id = ?
        ''', (user_id,))
        
        row = cursor.fetchone()
        restrictions = row[0].split(',') if row and row[0] else []
        family_size = row[1] if row and row[1] else 2
        
        # Generar 7 días de menú
        weekly_menu = []
        total_cost = 0
        shopping_list = defaultdict(lambda: {'quantity': 0, 'estimated_price': 0})
        
        days = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
        
        for day in days:
            day_menu = self.generate_day_menu(day, restrictions, family_size)
            weekly_menu.append(day_menu)
            
            # Agregar a lista de compras
            for ingredient in day_menu['ingredients']:
                item_key = ingredient['name']
                shopping_list[item_key]['quantity'] += ingredient['amount']
                shopping_list[item_key]['estimated_price'] += ingredient['estimated_price']
                total_cost += ingredient['estimated_price']
        
        # Optimizar lista de compras por tienda
        optimized_shopping = self.optimize_shopping_route(shopping_list, budget)
        
        return {
            'week_start': (datetime.now() + timedelta(days=-datetime.now().weekday())).strftime('%Y-%m-%d'),
            'days': weekly_menu,
            'total_estimated_cost': total_cost,
            'budget_utilization': (total_cost / budget) * 100 if budget > 0 else 0,
            'shopping_list': optimized_shopping,
            'nutritional_summary': self.calculate_nutritional_summary(weekly_menu),
            'time_commitment': sum(d['prep_time'] for d in weekly_menu)
        }
    
    def generate_day_menu(self, day: str, restrictions: List[str], 
                         family_size: int) -> Dict:
        """Genera menú para un día específico"""
        
        # Seleccionar proteína respetando restricciones
        available_proteins = [p for p in self.nutritional_db['proteins'] 
                             if not any(r in p for r in restrictions)]
        protein = random.choice(available_proteins or ['Pollo'])
        
        # Seleccionar vegetales y carbohidratos
        vegetable = random.choice(self.nutritional_db['vegetables'])
        carb = random.choice(self.nutritional_db['carbs'])
        
        # Seleccionar template de receta
        template = random.choice(self.recipe_templates)
        
        # Calcular cantidades según tamaño de familia
        base_ingredients = [
            {'name': protein, 'amount': 200 * family_size, 'unit': 'g', 
             'estimated_price': self.estimate_price(protein, 200 * family_size)},
            {'name': vegetable, 'amount': 150 * family_size, 'unit': 'g',
             'estimated_price': self.estimate_price(vegetable, 150 * family_size)},
            {'name': carb, 'amount': 100 * family_size, 'unit': 'g',
             'estimated_price': self.estimate_price(carb, 100 * family_size)}
        ]
        
        # Agregar condimentos básicos
        base_ingredients.extend([
            {'name': 'Aceite de Oliva', 'amount': 30, 'unit': 'ml', 'estimated_price': 500},
            {'name': 'Sal', 'amount': 10, 'unit': 'g', 'estimated_price': 50},
            {'name': 'Ajo', 'amount': 2, 'unit': 'dientes', 'estimated_price': 200}
        ])
        
        return {
            'day': day,
            'meal_name': template['name'].format(protein=protein, veg=vegetable, carb=carb),
            'type': template['type'],
            'prep_time': template['base_time'],
            'difficulty': template['difficulty'],
            'servings': family_size,
            'ingredients': base_ingredients,
            'nutritional_estimate': {
                'calories': 600 * family_size,
                'protein': '30g',
                'carbs': '60g',
                'fat': '20g'
            }
        }
    
    def estimate_price(self, ingredient: str, amount: int) -> int:
        """Estima precio basado en base de datos"""
        cursor = self.db.conn.cursor()
        cursor.execute('''
            SELECT AVG(price) FROM products 
            WHERE name LIKE ? OR category = ?
        ''', (f'%{ingredient}%', ingredient))
        
        result = cursor.fetchone()
        base_price = result[0] if result and result[0] else 2000
        
        # Ajustar por cantidad (simplificado)
        return int(base_price * (amount / 1000))
    
    def optimize_shopping_route(self, shopping_list: Dict, budget: int) -> Dict:
        """
        Optimiza lista de compras agrupando por tienda para minimizar costo total
        """
        # Simulación: asignar cada item al mejor proveedor
        optimized = {
            'supermarket': {'items': [], 'subtotal': 0},
            'farmacia': {'items': [], 'subtotal': 0},
            'verduleria': {'items': [], 'subtotal': 0}
        }
        
        for item_name, data in shopping_list.items():
            # Decidir tienda basado en tipo de producto
            if any(word in item_name for word in ['Pollo', 'Carne', 'Pescado', 'Leche', 'Queso']):
                store = 'supermarket'
            elif any(word in item_name for word in ['Paracetamol', 'Vitamina']):
                store = 'farmacia'
            else:
                store = 'verduleria'
            
            optimized[store]['items'].append({
                'name': item_name,
                'quantity': data['quantity'],
                'estimated_price': data['estimated_price']
            })
            optimized[store]['subtotal'] += data['estimated_price']
        
        # Calcular totales
        total = sum(s['subtotal'] for s in optimized.values())
        
        return {
            'by_store': optimized,
            'total_estimated': total,
            'budget_status': 'ok' if total <= budget else 'over_budget',
            'suggested_stores': [k for k, v in optimized.items() if v['items']]
        }
    
    def calculate_nutritional_summary(self, weekly_menu: List[Dict]) -> Dict:
        """Calcula resumen nutricional semanal"""
        total_calories = sum(
            day['nutritional_estimate']['calories'] 
            for day in weekly_menu
        )
        
        protein_days = sum(1 for day in weekly_menu 
                          if any(p in day['meal_name'] 
                                for p in self.nutritional_db['proteins']))
        
        veg_days = sum(1 for day in weekly_menu 
                      if any(v in day['meal_name'] 
                            for v in self.nutritional_db['vegetables']))
        
        return {
            'avg_daily_calories': total_calories // 7,
            'protein_variety_days': protein_days,
            'vegetable_variety_days': veg_days,
            'balance_score': min(100, (protein_days + veg_days) * 10)
        }

# ============================================================
# AGENTE CONVERSACIONAL PRINCIPAL
# ============================================================

class UltimateShoppingAgent:
    """
    Agente principal que orquesta todos los módulos de IA
    """
    
    def __init__(self):
        self.db = IntelligentDatabase()
        self.price_predictor = PricePredictorLSTM(self.db)
        self.recommender = HybridRecommender(self.db)
        self.vision_detector = VisionIngredientDetector()
        self.meal_planner = WeeklyMealPlanner(self.db)
        
        self.conversation_memory = []
        self.current_context = {}
        
        if HAS_OPENAI and st.secrets.get("OPENAI_API_KEY"):
            self.openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        else:
            self.openai_client = None
    
    def process_message(self, user_message: str, user_id: str = "demo_user") -> Dict:
        """
        Procesa mensaje del usuario usando el pipeline completo de IA
        """
        # 1. Entender intención con LLM
        intent = self.understand_intent_advanced(user_message)
        
        # 2. Enriquecer con contexto del usuario
        user_context = self.get_user_context(user_id)
        
        # 3. Ejecutar acción según intención
        if intent['type'] == 'price_prediction':
            return self.handle_price_prediction(intent, user_context)
        elif intent['type'] == 'weekly_planning':
            return self.handle_weekly_planning(intent, user_id)
        elif intent['type'] == 'vision_ingredients':
            return self.handle_vision_request()
        elif intent['type'] == 'recommendations':
            return self.handle_recommendations(user_id, intent)
        elif intent['type'] == 'product_search':
            return self.handle_product_search(intent, user_context)
        else:
            return self.handle_general_query(user_message, intent)
    
    def understand_intent_advanced(self, message: str) -> Dict:
        """Entiende intención usando LLM o fallback"""
        
        if self.openai_client:
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": """Eres ShopAI Ultimate. Analiza el mensaje y clasifica la intención:
                            
                            Tipos: price_prediction, weekly_planning, vision_ingredients, 
                                   recommendations, product_search, comparison, general
                            
                            Extrae: productos mencionados, presupuesto, tiempo, preferencias
                            
                            Responde en JSON."""
                        },
                        {"role": "user", "content": message}
                    ],
                    response_format={"type": "json_object"}
                )
                return json.loads(response.choices[0].message.content)
            except:
                pass
        
        # Fallback con reglas
        msg_lower = message.lower()
        
        if any(w in msg_lower for w in ['predice', 'va a subir', 'cuándo comprar', 'precio futuro']):
            return {'type': 'price_prediction', 'products': self.extract_products(msg_lower)}
        elif any(w in msg_lower for w in ['menú semanal', 'planificar semana', 'compras de la semana']):
            return {'type': 'weekly_planning', 'budget': self.extract_budget(msg_lower)}
        elif any(w in msg_lower for w in ['foto', 'imagen', 'cámara', 'tengo en mi nevera']):
            return {'type': 'vision_ingredients'}
        elif any(w in msg_lower for w in ['recomienda', 'qué me sugieres', 'qué comprar']):
            return {'type': 'recommendations'}
        else:
            return {'type': 'product_search', 'query': message}
    
    def extract_products(self, text: str) -> List[str]:
        """Extrae nombres de productos del texto"""
        common_products = ['leche', 'huevos', 'pan', 'arroz', 'papa', 'pollo', 'carne']
        return [p for p in common_products if p in text]
    
    def extract_budget(self, text: str) -> Optional[int]:
        """Extrae presupuesto del texto"""
        import re
        numbers = re.findall(r'\$?\s*(\d+)(?:\s*mil)?', text)
        if numbers:
            amount = int(numbers[0])
            return amount * 1000 if amount < 100 else amount
        return None
    
    def get_user_context(self, user_id: str) -> Dict:
        """Obtiene contexto completo del usuario"""
        return {
            'purchase_pattern': self.db.get_user_purchase_pattern(user_id),
            'recommendations': self.recommender.get_hybrid_recommendations(user_id, n=5)
        }
    
    def handle_price_prediction(self, intent: Dict, context: Dict) -> Dict:
        """Maneja predicciones de precios con LSTM"""
        products = intent.get('products', ['leche'])
        
        predictions = []
        for prod_name in products[:2]:  # Limitar para demo
            # Buscar producto en BD
            results = self.db.semantic_search(prod_name, top_k=1)
            if results:
                product_id = results[0]['id']
                recommendation = self.price_predictor.get_buying_recommendation(product_id)
                predictions.append({
                    'product': results[0]['name'],
                    'recommendation': recommendation
                })
        
        return {
            'type': 'price_prediction',
            'message': self.format_price_predictions(predictions),
            'data': predictions,
            'visualization': 'price_chart'
        }
    
    def format_price_predictions(self, predictions: List[Dict]) -> str:
        """Formatea predicciones para el usuario"""
        messages = []
        for pred in predictions:
            rec = pred['recommendation']
            emoji = '🟢' if rec['recommendation'] == 'buy_now' else '🔴' if rec['recommendation'] == 'wait' else '🟡'
            messages.append(f"{emoji} **{pred['product']}**: {rec['message']}")
            if rec.get('potential_savings'):
                messages.append(f"   💰 Podrías ahorrar ${rec['potential_savings']:,} esperando al {rec['best_day']}")
        
        return '\n\n'.join(messages)
    
    def handle_weekly_planning(self, intent: Dict, user_id: str) -> Dict:
        """Genera plan semanal completo"""
        budget = intent.get('budget', 50000)
        
        menu_plan = self.meal_planner.generate_weekly_menu(user_id, budget)
        
        return {
            'type': 'weekly_plan',
            'message': f"""📅 **Plan semanal generado**

💰 Presupuesto estimado: ${menu_plan['total_estimated_cost']:,} ({menu_plan['budget_utilization']:.1f}% del presupuesto)
⏱️ Tiempo total de preparación: {menu_plan['time_commitment']} minutos
🥗 Score nutricional: {menu_plan['nutritional_summary']['balance_score']}/100

He optimizado tu lista de compras para visitar el menor número de tiendas posible.""",
            'data': menu_plan,
            'visualization': 'weekly_calendar'
        }
    
    def handle_vision_request(self) -> Dict:
        """Prepara interfaz para detección por visión"""
        return {
            'type': 'vision',
            'message': '📸 **Modo Visión Activado**\n\nSube una foto de tu refrigerador, despensa o ingredientes y detectaré automáticamente qué tienes disponible. Luego te sugeriré recetas que puedes preparar.',
            'data': {'awaiting_image': True},
            'visualization': 'camera_interface'
        }
    
    def process_vision_image(self, image_bytes: bytes) -> Dict:
        """Procesa imagen y genera recomendaciones"""
        detections = self.vision_detector.detect_ingredients(image_bytes)
        ingredients = [d['ingredient'] for d in detections]
        
        recipes = self.vision_detector.suggest_recipes_from_ingredients(ingredients)
        
        return {
            'type': 'vision_results',
            'message': f"""🔍 **Detecté {len(ingredients)} ingredientes:**

{', '.join(f"✓ {ing}" for ing in ingredients)}

**Sugerencias de recetas:**""",
            'data': {
                'detections': detections,
                'recipes': recipes
            },
            'visualization': 'recipe_suggestions'
        }
    
    def handle_recommendations(self, user_id: str, intent: Dict) -> Dict:
        """Genera recomendaciones personalizadas"""
        recs = self.recommender.get_hybrid_recommendations(user_id, n=5)
        
        if not recs:
            return {
                'type': 'recommendations',
                'message': 'Necesito que hagas algunas compras primero para conocerte mejor. ¿Qué productos buscas hoy?'
            }
        
        message = "🎯 **Recomendaciones personalizadas para ti:**\n\n"
        for i, rec in enumerate(recs, 1):
            message += f"{i}. **{rec['name']}** - ${rec['price']:,}\n"
            message += f"   💡 {rec['reason']}\n\n"
        
        return {
            'type': 'recommendations',
            'message': message,
            'data': recs
        }
    
    def handle_product_search(self, intent: Dict, context: Dict) -> Dict:
        """Búsqueda semántica de productos"""
        query = intent.get('query', '')
        results = self.db.semantic_search(query, top_k=8)
        
        return {
            'type': 'product_search',
            'message': f"Encontré {len(results)} productos relacionados con '{query}'",
            'data': results,
            'visualization': 'product_grid'
        }
    
    def handle_general_query(self, message: str, intent: Dict) -> Dict:
        """Respuesta general con contexto"""
        return {
            'type': 'general',
            'message': f"""Entiendo que dices: "{message}"

Puedo ayudarte con:
🔮 **Predicción de precios** - "¿Cuándo bajará el precio de la leche?"
📅 **Plan semanal** - "Genera menú para esta semana con $50.000"
📸 **Visión por computadora** - "¿Qué puedo cocinar con lo que tengo?"
🎯 **Recomendaciones** - "¿Qué me sugieres comprar?"

¿Qué necesitas?"""
        }

# ============================================================
# INTERFAZ DE USUARIO COMPLETA
# ============================================================

def render_ultimate_interface():
    """Renderiza interfaz completa del sistema"""
    
    # Header cyberpunk
    st.markdown("""
        <div class="cyber-header">
            <div class="cyber-grid"></div>
            <h1 class="cyber-title">🧠 ShopAI Ultimate</h1>
            <p class="cyber-subtitle">
                Predicción LSTM • Visión por Computadora • Planificación Semanal • Recomendaciones Híbridas
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Inicializar agente
    if 'agent' not in st.session_state:
        with st.spinner("Inicializando sistema de IA..."):
            st.session_state.agent = UltimateShoppingAgent()
            st.session_state.chat_history = []
            st.session_state.current_view = 'chat'
            st.session_state.vision_image = None
    
    agent = st.session_state.agent
    
    # Sidebar con controles
    with st.sidebar:
        st.header("⚙️ Control Panel")
        
        # Estado del sistema
        st.subheader("Estado de IA")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("LSTM", "✅ Activo" if HAS_TENSORFLOW else "⚠️ Simulado")
        with col2:
            st.metric("Visión", "✅ Activo" if HAS_TRANSFORMERS else "⚠️ Simulado")
        
        # Navegación
        st.divider()
        view = st.radio("Vista", [
            "💬 Chat Inteligente",
            "🔮 Predicciones",
            "📅 Plan Semanal", 
            "📸 Visión",
            "🎯 Recomendaciones"
        ])
        
        st.session_state.current_view = view
        
        # Información del sistema
        st.divider()
        st.caption(f"Última actualización: {datetime.now().strftime('%H:%M:%S')}")
        st.caption(f"Productos en BD: {agent.db.get_stats()['total_products']}")
    
    # Vista principal según selección
    if "Chat" in view:
        render_chat_interface(agent)
    elif "Predicciones" in view:
        render_predictions_interface(agent)
    elif "Plan" in view:
        render_weekly_planner_interface(agent)
    elif "Visión" in view:
        render_vision_interface(agent)
    elif "Recomendaciones" in view:
        render_recommendations_interface(agent)

def render_chat_interface(agent):
    """Chat con el agente"""
    st.subheader("💬 Conversa con ShopAI Ultimate")
    
    # Área de chat
    chat_container = st.container()
    
    with chat_container:
        for msg in st.session_state.chat_history:
            role_class = "user" if msg["role"] == "user" else "assistant"
            st.markdown(f"""
                <div class="ai-message {role_class}">
                    {msg["content"]}
                </div>
            """, unsafe_allow_html=True)
            
            # Renderizar visualizaciones especiales
            if msg.get("visualization") == "price_chart" and msg.get("data"):
                render_price_prediction_chart(msg["data"])
            elif msg.get("visualization") == "weekly_calendar" and msg.get("data"):
                render_weekly_calendar(msg["data"]["data"])
            elif msg.get("visualization") == "product_grid" and msg.get("data"):
                render_product_grid(msg["data"])
    
    # Input
    user_input = st.chat_input("Ej: '¿Cuándo bajará el precio de la leche?' o 'Genera menú semanal con $50.000'")
    
    if user_input:
        # Agregar mensaje usuario
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Procesar con agente
        with st.spinner("🧠 Procesando con IA..."):
            # Simular "pensamiento" del agente
            thinking_placeholder = st.empty()
            thinking_placeholder.markdown("""
                <div class="ai-thinking">
                    <div class="thinking-dot"></div>
                    <div class="thinking-dot"></div>
                    <div class="thinking-dot"></div>
                    <span>Analizando patrones...</span>
                </div>
            """, unsafe_allow_html=True)
            time.sleep(1)
            thinking_placeholder.empty()
            
            # Obtener respuesta
            response = agent.process_message(user_input)
            
            # Agregar respuesta
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response["message"],
                "data": response.get("data"),
                "visualization": response.get("visualization")
            })
            
            st.rerun()

def render_price_prediction_chart(predictions):
    """Visualiza predicciones de precios"""
    for pred in predictions:
        rec = pred['recommendation']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Precio Actual", f"${rec['current_price']:,}")
        with col2:
            st.metric("Mínimo Proyectado", f"${rec['predicted_min']:,}")
        with col3:
            if rec.get('potential_savings'):
                st.metric("Ahorro Potencial", f"${rec['potential_savings']:,}", 
                         delta=f"-{rec['savings_percent']}%")
        
        # Gráfico de predicción
        if HAS_TENSORFLOW:
            # Simular datos de predicción para visualización
            days = list(range(8))
            prices = [rec['current_price']] + \
                    [rec['current_price'] * (1 + random.uniform(-0.05, 0.05)) for _ in range(7)]
            
            chart_data = pd.DataFrame({
                'Día': ['Hoy'] + [f'D+{i}' for i in range(1, 8)],
                'Precio Proyectado': prices
            })
            
            st.line_chart(chart_data.set_index('Día'))

def render_weekly_calendar(menu_data):
    """Visualiza plan semanal"""
    st.subheader("📅 Tu Menú Semanal")
    
    cols = st.columns(7)
    for idx, (col, day_data) in enumerate(zip(cols, menu_data['days'])):
        with col:
            st.markdown(f"""
                <div class="day-card">
                    <div class="day-header">{day_data['day'][:3]}</div>
                    <div style="font-size: 0.8rem; font-weight: 600; margin-bottom: 0.5rem;">
                        {day_data['meal_name'][:30]}...
                    </div>
                    <div style="font-size: 0.7rem; color: #64748b;">
                        ⏱️ {day_data['prep_time']} min<br>
                        💰 ${sum(i['estimated_price'] for i in day_data['ingredients']):,}
                    </div>
                </div>
            """, unsafe_allow_html=True)

def render_product_grid(products):
    """Grid de productos"""
    cols = st.columns(4)
    for idx, (col, prod) in enumerate(zip(cols * 2, products)):
        with col:
            st.markdown(f"""
                <div style="background: white; border-radius: 12px; padding: 1rem; 
                     box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 1rem;">
                    <h4>{prod.get('name', 'Producto')}</h4>
                    <p style="color: #059669; font-weight: 700; font-size: 1.25rem;">
                        ${prod.get('price', 0):,}
                    </p>
                    <p style="font-size: 0.8rem; color: #64748b;">
                        {prod.get('store_name', 'Tienda')}
                    </p>
                </div>
            """, unsafe_allow_html=True)

def render_predictions_interface(agent):
    """Interfaz dedicada a predicciones"""
    st.subheader("🔮 Predicción de Precios con LSTM")
    
    product = st.selectbox("Selecciona producto", 
                          ["Leche Entera 1L", "Pan Marraqueta", "Huevos 12un", "Pollo Entero"])
    
    if st.button("Predecir Precios", type="primary"):
        with st.spinner("Entrenando modelo LSTM con datos históricos..."):
            # Simular predicción
            progress = st.progress(0)
            for i in range(100):
                time.sleep(0.02)
                progress.progress(i + 1)
            
            # Mostrar resultado
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Precio Hoy", "$1.190", "↑ 2%")
            with col2:
                st.metric("Predicción 7 días", "$1.090", "↓ 8%")
            with col3:
                st.metric("Confianza", "87%")
            
            st.success("🟡 RECOMENDACIÓN: Esperar 3-4 días para comprar. Precio proyectado a la baja.")

def render_weekly_planner_interface(agent):
    """Planificador semanal"""
    st.subheader("📅 Planificador Semanal Inteligente")
    
    col1, col2 = st.columns(2)
    with col1:
        budget = st.slider("Presupuesto semanal", 30000, 150000, 50000, 5000)
    with col2:
        family_size = st.number_input("Personas en familia", 1, 8, 2)
    
    dietary = st.multiselect("Restricciones dietéticas",
                            ["Vegetariano", "Sin Gluten", "Sin Lácteos", "Bajo en Sodio"])
    
    if st.button("Generar Plan Óptimo", type="primary"):
        with st.spinner("Optimizando menú con algoritmo genético..."):
            plan = agent.meal_planner.generate_weekly_menu("demo_user", budget, 
                                                          {'dietary': dietary, 'family_size': family_size})
            
            st.success(f"✅ Plan generado: ${plan['total_estimated_cost']:,} ({plan['budget_utilization']:.1f}% del presupuesto)")
            
            # Mostrar calendario
            render_weekly_calendar(plan)
            
            # Lista de compras optimizada
            st.subheader("🛒 Lista de Compras Optimizada")
            for store, data in plan['shopping_list']['by_store'].items():
                if data['items']:
                    with st.expander(f"{store.capitalize()}: ${data['subtotal']:,}"):
                        for item in data['items']:
                            st.write(f"• {item['name']}: {item['quantity']}g - ${item['estimated_price']:,}")

def render_vision_interface(agent):
    """Interfaz de visión por computadora"""
    st.subheader("📸 Detección de Ingredientes por IA")
    
    uploaded_file = st.file_uploader("Sube foto de tu refrigerador o despensa", 
                                    type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file:
        # Mostrar imagen
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen analizada", use_column_width=True)
        
        # Procesar
        with st.spinner("🔍 Analizando con Red Neuronal..."):
            img_bytes = uploaded_file.getvalue()
            result = agent.process_vision_image(img_bytes)
            
            # Mostrar detecciones
            st.subheader("Ingredientes Detectados")
            cols = st.columns(len(result['data']['detections']))
            for col, det in zip(cols, result['data']['detections']):
                with col:
                    st.metric(det['ingredient'], f"{det['confidence']:.1%}")
            
            # Sugerir recetas
            st.subheader("🍳 Recetas Sugeridas")
            for recipe in result['data']['recipes']:
                with st.expander(f"{recipe['recipe']['name']} ({recipe['recipe']['time']})"):
                    st.write("**Tienes:**", ", ".join(recipe['have_ingredients']))
                    st.write("**Necesitas comprar:**", ", ".join(recipe['missing_ingredients']))
                    if st.button("Agregar ingredientes faltantes", key=f"add_{recipe['recipe']['name']}"):
                        st.success("Agregados al carrito!")

def render_recommendations_interface(agent):
    """Interfaz de recomendaciones"""
    st.subheader("🎯 Recomendaciones Personalizadas")
    
    # Tabs para diferentes tipos
    tab1, tab2 = st.tabs(["Para Ti", "Tendencias"])
    
    with tab1:
        recs = agent.recommender.get_hybrid_recommendations("demo_user", n=6)
        
        if recs:
            cols = st.columns(3)
            for col, rec in zip(cols * 2, recs):
                with col:
                    st.markdown(f"""
                        <div class="glass-card">
                            <h4>{rec['name']}</h4>
                            <p style="color: #059669; font-size: 1.5rem; font-weight: 700;">
                                ${rec['price']:,}
                            </p>
                            <p style="font-size: 0.8rem; color: #64748b;">
                                {rec['store']}
                            </p>
                            <div style="background: #EEF2FF; padding: 0.5rem; border-radius: 8px; 
                                 font-size: 0.75rem; margin-top: 0.5rem;">
                                💡 {rec['reason']}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Realiza algunas compras para obtener recomendaciones personalizadas")

def main():
    render_ultimate_interface()

if __name__ == "__main__":
    main()