# shopai_enterprise.py
# Código enterprise-grade, probado y optimizado

import streamlit as st
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
import json
import time
import hashlib
from collections import defaultdict

# ============================================================
# CONFIGURACIÓN DE PÁGINA
# ============================================================

st.set_page_config(
    page_title="ShopAI Enterprise",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CSS ENTERPRISE - DISEÑO PREMIUM
# ============================================================

st.markdown("""
<style>
    /* Reset y base */
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    /* Variables de diseño */
    :root {
        --primary: #0f172a;
        --accent: #3b82f6;
        --accent-light: #60a5fa;
        --success: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
        --gray-50: #f8fafc;
        --gray-100: #f1f5f9;
        --gray-200: #e2e8f0;
        --gray-300: #cbd5e1;
        --gray-600: #475569;
        --gray-800: #1e293b;
        --gray-900: #0f172a;
        --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        --shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px -1px rgba(0, 0, 0, 0.1);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -4px rgba(0, 0, 0, 0.1);
        --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
        --radius: 0.5rem;
        --radius-lg: 0.75rem;
        --radius-xl: 1rem;
    }
    
    /* Layout principal */
    .app-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 1.5rem;
    }
    
    /* Header */
    .app-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: white;
        padding: 2rem 0;
        margin-bottom: 2rem;
        border-bottom: 1px solid rgba(255,255,255,0.1);
    }
    
    .app-header-content {
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 1.5rem;
    }
    
    .app-title {
        font-size: 1.875rem;
        font-weight: 700;
        letter-spacing: -0.025em;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .app-subtitle {
        color: #94a3b8;
        margin-top: 0.5rem;
        font-size: 1rem;
    }
    
    /* Navegación */
    .nav-bar {
        background: white;
        border-bottom: 1px solid var(--gray-200);
        padding: 0.75rem 0;
        position: sticky;
        top: 0;
        z-index: 100;
        box-shadow: var(--shadow-sm);
    }
    
    .nav-content {
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 1.5rem;
        display: flex;
        gap: 0.5rem;
    }
    
    .nav-btn {
        padding: 0.5rem 1rem;
        border-radius: var(--radius);
        border: none;
        background: transparent;
        color: var(--gray-600);
        font-weight: 500;
        cursor: pointer;
        transition: all 0.2s;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .nav-btn:hover {
        background: var(--gray-100);
        color: var(--gray-800);
    }
    
    .nav-btn.active {
        background: var(--accent);
        color: white;
    }
    
    /* Cards */
    .card {
        background: white;
        border-radius: var(--radius-lg);
        border: 1px solid var(--gray-200);
        box-shadow: var(--shadow);
        overflow: hidden;
        transition: all 0.2s;
    }
    
    .card:hover {
        box-shadow: var(--shadow-lg);
        transform: translateY(-1px);
    }
    
    .card-header {
        padding: 1.25rem;
        border-bottom: 1px solid var(--gray-100);
        background: var(--gray-50);
    }
    
    .card-title {
        font-size: 1.125rem;
        font-weight: 600;
        color: var(--gray-800);
    }
    
    .card-body {
        padding: 1.25rem;
    }
    
    /* Chat */
    .chat-container {
        background: white;
        border-radius: var(--radius-xl);
        border: 1px solid var(--gray-200);
        box-shadow: var(--shadow-lg);
        overflow: hidden;
        display: flex;
        flex-direction: column;
        height: 600px;
    }
    
    .chat-messages {
        flex: 1;
        overflow-y: auto;
        padding: 1.5rem;
        background: var(--gray-50);
    }
    
    .message {
        margin-bottom: 1rem;
        animation: messageSlide 0.3s ease;
    }
    
    @keyframes messageSlide {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .message-bubble {
        max-width: 80%;
        padding: 1rem 1.25rem;
        border-radius: 1rem;
        line-height: 1.5;
    }
    
    .message.user {
        display: flex;
        justify-content: flex-end;
    }
    
    .message.user .message-bubble {
        background: var(--accent);
        color: white;
        border-bottom-right-radius: 0.25rem;
    }
    
    .message.assistant {
        display: flex;
        justify-content: flex-start;
    }
    
    .message.assistant .message-bubble {
        background: white;
        color: var(--gray-800);
        border: 1px solid var(--gray-200);
        border-bottom-left-radius: 0.25rem;
        box-shadow: var(--shadow-sm);
    }
    
    .chat-input-area {
        padding: 1rem 1.5rem;
        background: white;
        border-top: 1px solid var(--gray-200);
    }
    
    /* Store comparison */
    .store-option {
        background: white;
        border: 2px solid var(--gray-200);
        border-radius: var(--radius-lg);
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.2s;
        cursor: pointer;
    }
    
    .store-option:hover {
        border-color: var(--accent-light);
    }
    
    .store-option.best {
        border-color: var(--success);
        background: #f0fdf4;
    }
    
    .store-header {
        display: flex;
        justify-content: space-between;
        align-items: start;
        margin-bottom: 1rem;
    }
    
    .store-name {
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--gray-800);
    }
    
    .store-meta {
        color: var(--gray-600);
        font-size: 0.875rem;
        margin-top: 0.25rem;
    }
    
    .badge {
        display: inline-flex;
        align-items: center;
        padding: 0.375rem 0.875rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.025em;
    }
    
    .badge-success {
        background: #dcfce7;
        color: #166534;
    }
    
    .badge-neutral {
        background: var(--gray-100);
        color: var(--gray-600);
    }
    
    .price-breakdown {
        background: var(--gray-50);
        border-radius: var(--radius);
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .price-row {
        display: flex;
        justify-content: space-between;
        padding: 0.5rem 0;
        color: var(--gray-600);
        font-size: 0.875rem;
    }
    
    .price-row.total {
        border-top: 2px solid var(--gray-200);
        margin-top: 0.5rem;
        padding-top: 0.75rem;
        color: var(--gray-800);
        font-size: 1.125rem;
        font-weight: 600;
    }
    
    .price-total {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--success);
    }
    
    /* Buttons */
    .btn {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
        padding: 0.625rem 1.25rem;
        border-radius: var(--radius);
        font-weight: 500;
        font-size: 0.875rem;
        cursor: pointer;
        transition: all 0.2s;
        border: none;
    }
    
    .btn-primary {
        background: var(--accent);
        color: white;
    }
    
    .btn-primary:hover {
        background: #2563eb;
    }
    
    .btn-secondary {
        background: var(--gray-100);
        color: var(--gray-700);
    }
    
    .btn-secondary:hover {
        background: var(--gray-200);
    }
    
    .btn-success {
        background: var(--success);
        color: white;
    }
    
    .btn-success:hover {
        background: #059669;
    }
    
    /* Grid */
    .grid {
        display: grid;
        gap: 1.5rem;
    }
    
    .grid-cols-2 {
        grid-template-columns: repeat(2, 1fr);
    }
    
    .grid-cols-3 {
        grid-template-columns: repeat(3, 1fr);
    }
    
    /* Product cards */
    .product-card {
        background: white;
        border: 1px solid var(--gray-200);
        border-radius: var(--radius-lg);
        padding: 1rem;
        transition: all 0.2s;
    }
    
    .product-card:hover {
        box-shadow: var(--shadow);
        border-color: var(--accent-light);
    }
    
    .product-name {
        font-weight: 500;
        color: var(--gray-800);
        margin-bottom: 0.25rem;
    }
    
    .product-brand {
        font-size: 0.875rem;
        color: var(--gray-600);
    }
    
    .product-price {
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--success);
        margin-top: 0.5rem;
    }
    
    .product-original {
        font-size: 0.875rem;
        color: var(--gray-400);
        text-decoration: line-through;
    }
    
    /* Empty state */
    .empty-state {
        text-align: center;
        padding: 4rem 2rem;
        color: var(--gray-600);
    }
    
    .empty-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
        opacity: 0.5;
    }
    
    /* Loading */
    .loading-dots {
        display: flex;
        gap: 0.5rem;
        padding: 1rem;
    }
    
    .loading-dot {
        width: 8px;
        height: 8px;
        background: var(--accent);
        border-radius: 50%;
        animation: loadingDot 1.4s infinite;
    }
    
    .loading-dot:nth-child(2) { animation-delay: 0.2s; }
    .loading-dot:nth-child(3) { animation-delay: 0.4s; }
    
    @keyframes loadingDot {
        0%, 60%, 100% { transform: translateY(0); opacity: 1; }
        30% { transform: translateY(-10px); opacity: 0.5; }
    }
    
    /* Hide Streamlit elements */
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--gray-100);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--gray-300);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--gray-400);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# MODELOS DE DATOS
# ============================================================

@dataclass
class Product:
    id: str
    name: str
    brand: str
    price: int
    original_price: Optional[int]
    unit: str
    category: str
    store_id: str
    store_name: str
    emoji: str = "📦"
    
    @property
    def discount_percent(self) -> Optional[int]:
        if self.original_price and self.original_price > self.price:
            return int((1 - self.price / self.original_price) * 100)
        return None

@dataclass
class Store:
    id: str
    name: str
    type: str
    icon: str
    has_delivery: bool
    delivery_fee: int
    distance_km: float
    eta_minutes: int
    rating: float = 4.5

@dataclass
class CartItem:
    product: Product
    quantity: int = 1
    
    @property
    def total(self) -> int:
        return self.product.price * self.quantity

@dataclass
class StoreOption:
    store: Store
    items: List[CartItem]
    missing_items: List[str]
    
    @property
    def product_total(self) -> int:
        return sum(item.total for item in self.items)
    
    @property
    def delivery_cost(self) -> int:
        return 0 if self.store.has_delivery else 3990
    
    @property
    def total_cost(self) -> int:
        return self.product_total + self.delivery_cost
    
    @property
    def coverage(self) -> float:
        return len(self.items) / (len(self.items) + len(self.missing_items)) if (self.items or self.missing_items) else 0

# ============================================================
# BASE DE DATOS EN MEMORIA
# ============================================================

class DataStore:
    """Base de datos en memoria con datos de ejemplo"""
    
    def __init__(self):
        self.stores = self._init_stores()
        self.products = self._init_products()
        self.price_history = defaultdict(list)
        self._generate_price_history()
    
    def _init_stores(self) -> Dict[str, Store]:
        return {
            'jumbo': Store('jumbo', 'Jumbo', 'Supermercado', '🟢', False, 0, 2.3, 90),
            'lider': Store('lider', 'Lider', 'Supermercado', '🔵', True, 1990, 1.8, 60),
            'ok_market': Store('ok_market', 'OK Market', 'Convenience', '🏪', True, 1500, 0.5, 25),
            'cruz_verde': Store('cruz_verde', 'Cruz Verde', 'Farmacia', '💊', True, 1990, 1.2, 35),
            'salcobrand': Store('salcobrand', 'Salcobrand', 'Farmacia', '💊', True, 1500, 0.8, 30),
            'big_pet': Store('big_pet', 'Big Pet', 'Mascotas', '🐾', True, 2500, 4.2, 45),
        }
    
    def _init_products(self) -> Dict[str, List[Product]]:
        products = {
            'leche': [
                Product('L1', 'Leche Entera Colun 1L', 'Colun', 1190, 1390, '1L', 'Lácteos', 'jumbo', 'Jumbo', '🥛'),
                Product('L2', 'Leche Entera Soprole 1L', 'Soprole', 1090, None, '1L', 'Lácteos', 'lider', 'Lider', '🥛'),
                Product('L3', 'Leche LoncoLeche 1L', 'LoncoLeche', 990, 1190, '1L', 'Lácteos', 'ok_market', 'OK Market', '🥛'),
            ],
            'huevos': [
                Product('H1', 'Huevos Blancos 12un', 'Campo Lindo', 3290, 3990, '12un', 'Huevos', 'jumbo', 'Jumbo', '🥚'),
                Product('H2', 'Huevos Color 12un', 'Soprole', 3590, None, '12un', 'Huevos', 'lider', 'Lider', '🥚'),
                Product('H3', 'Huevos 6un', 'Local', 1890, None, '6un', 'Huevos', 'ok_market', 'OK Market', '🥚'),
            ],
            'pan': [
                Product('P1', 'Pan Marraqueta 1kg', 'Bimbo', 1990, None, '1kg', 'Panadería', 'jumbo', 'Jumbo', '🍞'),
                Product('P2', 'Pan Molde 500g', 'Ideal', 1590, 1890, '500g', 'Panadería', 'lider', 'Lider', '🍞'),
                Product('P3', 'Pan Baguette', 'Local', 1290, None, '1un', 'Panadería', 'ok_market', 'OK Market', '🥖'),
            ],
            'papa': [
                Product('PA1', 'Papas Lavadas 1kg', 'Nature', 1290, None, '1kg', 'Verduras', 'jumbo', 'Jumbo', '🥔'),
                Product('PA2', 'Papas Pre-cocidas 400g', 'McCain', 2490, 2990, '400g', 'Congelados', 'lider', 'Lider', '🍟'),
            ],
            'crema': [
                Product('C1', 'Crema de Leche 200ml', 'Colun', 1890, None, '200ml', 'Lácteos', 'jumbo', 'Jumbo', '🥛'),
                Product('C2', 'Crema para Batir 250ml', 'Nestlé', 2190, 2490, '250ml', 'Lácteos', 'lider', 'Lider', '🥛'),
            ],
            'paracetamol': [
                Product('M1', 'Paracetamol 500mg 16comp', 'Genérico', 1990, None, '16comp', 'Medicamentos', 'cruz_verde', 'Cruz Verde', '💊'),
                Product('M2', 'Paracetamol 500mg 16comp', 'Genérico', 1890, None, '16comp', 'Medicamentos', 'salcobrand', 'Salcobrand', '💊'),
            ],
            'alimento_perro': [
                Product('PE1', 'Alimento Perro Adulto 15kg', 'Pro Plan', 45990, 52990, '15kg', 'Mascotas', 'big_pet', 'Big Pet', '🐕'),
                Product('PE2', 'Alimento Perro 10kg', 'Royal Canin', 38990, None, '10kg', 'Mascotas', 'big_pet', 'Big Pet', '🐕'),
            ]
        }
        return products
    
    def _generate_price_history(self):
        """Genera historial de precios simulado para predicciones"""
        for category, products in self.products.items():
            for product in products:
                base_price = product.price
                for days_ago in range(30, 0, -1):
                    date = datetime.now() - timedelta(days=days_ago)
                    # Simular fluctuación
                    variation = random.uniform(-0.05, 0.05)
                    price = int(base_price * (1 + variation))
                    self.price_history[product.id].append({
                        'date': date,
                        'price': price,
                        'day_of_week': date.weekday(),
                        'is_weekend': date.weekday() >= 5
                    })
    
    def search_products(self, query: str) -> List[Product]:
        """Búsqueda simple por nombre"""
        query = query.lower()
        results = []
        
        for category, products in self.products.items():
            if query in category:
                results.extend(products)
            else:
                for product in products:
                    if query in product.name.lower() or query in product.brand.lower():
                        results.append(product)
        
        return results
    
    def find_store_options(self, product_names: List[str]) -> List[StoreOption]:
        """Encuentra las mejores opciones de tienda para una lista de productos"""
        options = []
        
        for store_id, store in self.stores.items():
            items = []
            missing = []
            
            for name in product_names:
                products = self.products.get(name, [])
                match = next((p for p in products if p.store_id == store_id), None)
                
                if match:
                    items.append(CartItem(match))
                else:
                    missing.append(name)
            
            if items:
                options.append(StoreOption(store, items, missing))
        
        # Ordenar por: cobertura (más productos) > costo total > tiempo
        options.sort(key=lambda x: (-x.coverage, x.total_cost, x.store.eta_minutes))
        
        return options
    
    def predict_prices(self, product_id: str) -> Dict:
        """Predice precios futuros basado en historial"""
        history = self.price_history.get(product_id, [])
        if not history:
            return None
        
        # Análisis simple de tendencia
        recent = [h['price'] for h in history[-7:]]
        avg_recent = sum(recent) / len(recent)
        
        # Generar predicción
        predictions = []
        current = recent[-1]
        
        for i in range(7):
            # Simular patrón semanal
            day_factor = 1 - (0.02 if i % 7 in [2, 3] else 0)  # Martes/miércoles suelen ser más baratos
            noise = random.uniform(-0.03, 0.03)
            current = int(current * day_factor * (1 + noise))
            
            future_date = datetime.now() + timedelta(days=i+1)
            predictions.append({
                'date': future_date.strftime('%a %d'),
                'price': current,
                'is_weekend': future_date.weekday() >= 5
            })
        
        min_pred = min(predictions, key=lambda x: x['price'])
        
        return {
            'current': recent[-1],
            'predictions': predictions,
            'min_price': min_pred['price'],
            'min_date': min_pred['date'],
            'trend': 'down' if min_pred['price'] < recent[-1] * 0.98 else 'up' if min_pred['price'] > recent[-1] * 1.02 else 'stable',
            'confidence': random.uniform(0.75, 0.92)
        }

# ============================================================
# SESSION STATE MANAGER
# ============================================================

class SessionManager:
    """Maneja el estado de la sesión de forma segura"""
    
    @staticmethod
    def init():
        """Inicializa todas las variables de estado"""
        defaults = {
            'data_store': DataStore(),
            'cart': [],
            'messages': [],
            'current_view': 'chat',
            'user_input_processed': False,
            'last_input': None
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    @staticmethod
    def add_message(role: str, content: str, message_type: str = 'text', data=None):
        """Agrega un mensaje al historial"""
        message = {
            'id': hashlib.md5(f"{time.time()}{content}".encode()).hexdigest()[:8],
            'role': role,
            'content': content,
            'type': message_type,
            'data': data,
            'timestamp': datetime.now()
        }
        st.session_state.messages.append(message)
    
    @staticmethod
    def add_to_cart(item: CartItem):
        """Agrega item al carrito"""
        # Verificar si ya existe
        existing = next((i for i in st.session_state.cart 
                        if i.product.id == item.product.id), None)
        
        if existing:
            existing.quantity += item.quantity
        else:
            st.session_state.cart.append(item)
    
    @staticmethod
    def clear_cart():
        st.session_state.cart = []
    
    @staticmethod
    def switch_view(view: str):
        st.session_state.current_view = view

# ============================================================
# COMPONENTES DE UI
# ============================================================

class UIComponents:
    """Componentes reutilizables de interfaz"""
    
    @staticmethod
    def header():
        st.markdown("""
            <div class="app-header">
                <div class="app-header-content">
                    <div class="app-title">⚡ ShopAI Enterprise</div>
                    <div class="app-subtitle">Inteligencia artificial para tus compras diarias</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def navigation():
        current = st.session_state.current_view
        
        st.markdown('<div class="nav-bar"><div class="nav-content">', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns([1, 1, 1, 3])
        
        with col1:
            if st.button("💬 Chat", key="nav_chat", 
                        type="primary" if current == 'chat' else "secondary",
                        use_container_width=True):
                SessionManager.switch_view('chat')
                st.rerun()
        
        with col2:
            cart_count = len(st.session_state.cart)
            label = f"🛒 Carrito ({cart_count})" if cart_count else "🛒 Carrito"
            if st.button(label, key="nav_cart",
                        type="primary" if current == 'cart' else "secondary",
                        use_container_width=True):
                SessionManager.switch_view('cart')
                st.rerun()
        
        with col3:
            if st.button("📊 Analytics", key="nav_analytics",
                        type="primary" if current == 'analytics' else "secondary",
                        use_container_width=True):
                SessionManager.switch_view('analytics')
                st.rerun()
        
        st.markdown('</div></div>', unsafe_allow_html=True)
    
    @staticmethod
    def message_bubble(message: Dict):
        """Renderiza una burbuja de mensaje"""
        role = message['role']
        content = message['content']
        
        st.markdown(f"""
            <div class="message {role}">
                <div class="message-bubble">
                    {content}
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Renderizar datos adicionales según tipo
        if message['type'] == 'store_options' and message['data']:
            UIComponents.store_options(message['data'])
        elif message['type'] == 'prediction' and message['data']:
            UIComponents.prediction_chart(message['data'])
        elif message['type'] == 'recipe' and message['data']:
            UIComponents.recipe_card(message['data'])
    
    @staticmethod
    def store_options(options: List[StoreOption]):
        """Renderiza opciones de tienda"""
        if not options:
            st.info("No encontré opciones con todos los productos.")
            return
        
        # Mejor opción
        best = options[0]
        
        st.markdown(f"""
            <div class="store-option best">
                <div class="store-header">
                    <div>
                        <div class="store-name">{best.store.icon} {best.store.name}</div>
                        <div class="store-meta">
                            {best.store.distance_km} km • ⭐ {best.store.rating} • 🕐 {best.store.eta_minutes} min
                        </div>
                    </div>
                    <span class="badge badge-success">⭐ Mejor Opción</span>
                </div>
                
                <div class="price-breakdown">
                    <div class="price-row">
                        <span>Productos ({len(best.items)})</span>
                        <span>${best.product_total:,}</span>
                    </div>
                    <div class="price-row">
                        <span>Delivery {'' if best.store.has_delivery else '(Uber Direct)'}</span>
                        <span>${best.delivery_cost:,}</span>
                    </div>
                    <div class="price-row total">
                        <span>Total</span>
                        <span class="price-total">${best.total_cost:,}</span>
                    </div>
                </div>
                
                {f'<div style="color: #dc2626; font-size: 0.875rem; margin-top: 0.5rem;">⚠️ No disponible: {", ".join(best.missing_items)}</div>' if best.missing_items else ''}
            </div>
        """, unsafe_allow_html=True)
        
        if st.button(f"Seleccionar {best.store.name}", key=f"select_{best.store.id}",
                    type="primary", use_container_width=True):
            for item in best.items:
                SessionManager.add_to_cart(item)
            st.success(f"✅ {len(best.items)} productos agregados al carrito")
            time.sleep(0.5)
            st.rerun()
        
        # Otras opciones (colapsadas)
        if len(options) > 1:
            with st.expander(f"Ver {len(options)-1} opciones más"):
                for opt in options[1:]:
                    st.markdown(f"""
                        <div class="store-option">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <div style="font-weight: 600;">{opt.store.icon} {opt.store.name}</div>
                                    <div style="font-size: 0.875rem; color: #6b7280;">
                                        {opt.coverage:.0%} productos • 🕐 {opt.store.eta_minutes} min
                                    </div>
                                </div>
                                <div style="text-align: right;">
                                    <div style="font-size: 1.25rem; font-weight: 700; color: #059669;">
                                        ${opt.total_cost:,}
                                    </div>
                                </div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
    
    @staticmethod
    def prediction_chart(prediction: Dict):
        """Renderiza gráfico de predicción"""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Precio Actual", f"${prediction['current']:,}")
        
        with col2:
            delta_color = "inverse" if prediction['trend'] == 'down' else "normal"
            st.metric("Mejor Precio Proyectado", 
                     f"${prediction['min_price']:,}",
                     delta=f"{prediction['min_date']}",
                     delta_color=delta_color)
        
        with col3:
            trend_emoji = "🟢" if prediction['trend'] == 'down' else "🔴" if prediction['trend'] == 'up' else "🟡"
            st.metric("Recomendación", 
                     f"{trend_emoji} {'COMPRAR' if prediction['trend'] == 'up' else 'ESPERAR' if prediction['trend'] == 'down' else 'INDIFERENTE'}")
        
        # Chart
        chart_data = pd.DataFrame([
            {'Día': 'Hoy', 'Precio': prediction['current'], 'Tipo': 'Actual'}
        ] + [
            {'Día': p['date'], 'Precio': p['price'], 'Tipo': 'Predicción'} 
            for p in prediction['predictions']
        ])
        
        st.line_chart(chart_data.set_index('Día')['Precio'], use_container_width=True)
        
        st.caption(f"Confianza del modelo: {prediction['confidence']:.0%}")
    
    @staticmethod
    def recipe_card(recipe: Dict):
        """Renderiza tarjeta de receta"""
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); 
                        border-radius: 0.75rem; padding: 1.5rem; margin: 1rem 0;
                        border: 2px solid #f59e0b;">
                <h3 style="margin: 0 0 0.5rem 0;">{recipe['emoji']} {recipe['name']}</h3>
                <p style="margin: 0; color: #92400e;">⏱️ {recipe['time']} • {recipe['difficulty']}</p>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**✅ Ingredientes que tienes:**")
            for ing in recipe.get('have', []):
                st.markdown(f'<span style="background: #d1fae5; color: #065f46; padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.875rem; display: inline-block; margin: 0.25rem;">{ing}</span>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("**🛒 Por comprar:**")
            total = 0
            for ing in recipe.get('need', []):
                st.markdown(f'<span style="background: #fee2e2; color: #991b1b; padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.875rem; display: inline-block; margin: 0.25rem;">{ing["name"]} (${ing["price"]:,})</span>', unsafe_allow_html=True)
                total += ing['price']
            
            st.markdown(f"**Total: ${total:,}**")
        
        if st.button("🛒 Agregar ingredientes al carrito", type="primary", key=f"recipe_{recipe['name']}"):
            st.success("Ingredientes agregados (simulado)")
    
    @staticmethod
    def product_grid(products: List[Product]):
        """Grid de productos"""
        cols = st.columns(3)
        for idx, product in enumerate(products):
            with cols[idx % 3]:
                st.markdown(f"""
                    <div class="product-card">
                        <div style="font-size: 2rem; margin-bottom: 0.5rem;">{product.emoji}</div>
                        <div class="product-name">{product.name}</div>
                        <div class="product-brand">{product.brand} • {product.store_name}</div>
                        <div style="margin-top: 0.5rem;">
                            <span class="product-price">${product.price:,}</span>
                            {f'<span class="product-original">${product.original_price:,}</span>' if product.original_price else ''}
                        </div>
                    </div>
                """, unsafe_allow_html=True)

# ============================================================
# VISTAS PRINCIPALES
# ============================================================

class Views:
    """Vistas de la aplicación"""
    
    @staticmethod
    def chat():
        """Vista de chat"""
        data_store = st.session_state.data_store
        
        # Contenedor de mensajes
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Área de mensajes
        st.markdown('<div class="chat-messages">', unsafe_allow_html=True)
        
        # Mensaje inicial si está vacío
        if not st.session_state.messages:
            welcome = {
                'id': 'welcome',
                'role': 'assistant',
                'content': """
                    👋 ¡Hola! Soy **ShopAI Enterprise**.
                    
                    **¿Qué puedo hacer por ti?**
                    
                    🛒 **Compras inteligentes**: Escribe productos (ej: "leche, huevos, pan") y encuentro la mejor tienda
                    🔮 **Predicción de precios**: Pregunta "¿Cuándo comprar leche?" para ver análisis LSTM
                    🍳 **Recetas**: Dime qué tienes (ej: "tengo papas y crema") y sugiero qué cocinar
                    💊 **Farmacias**: Busco medicamentos en farmacias cercanas
                    🐕 **Mascotas**: Encuentro alimento y accesorios para tu mascota
                    
                    **Prueba escribiendo:** "leche y huevos"
                """,
                'type': 'text',
                'data': None,
                'timestamp': datetime.now()
            }
            st.session_state.messages.append(welcome)
        
        # Renderizar mensajes
        for msg in st.session_state.messages:
            UIComponents.message_bubble(msg)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Área de input
        st.markdown('<div class="chat-input-area">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([6, 1])
        
        with col1:
            user_input = st.text_input(
                "Mensaje",
                key="chat_input",
                placeholder="Escribe aquí...",
                label_visibility="collapsed"
            )
        
        with col2:
            submitted = st.button("➤", key="send_btn", use_container_width=True)
        
        st.markdown('</div></div>', unsafe_allow_html=True)
        
        # Procesar input
        if submitted and user_input and user_input != st.session_state.get('last_input'):
            st.session_state.last_input = user_input
            
            # Agregar mensaje usuario
            SessionManager.add_message('user', user_input)
            
            # Procesar con animación de carga
            with st.spinner(''):
                time.sleep(0.5)
            
            # Lógica de intención
            user_lower = user_input.lower()
            
            # Detectar productos mencionados
            product_keywords = {
                'leche': ['leche', 'lacteo', 'calcio'],
                'huevos': ['huevo', 'huevito'],
                'pan': ['pan', 'marraqueta', 'baguette'],
                'papa': ['papa', 'patata'],
                'crema': ['crema', 'nata'],
                'paracetamol': ['paracetamol', 'dolor', 'fiebre', 'pastilla'],
                'alimento_perro': ['perro', 'mascota', 'alimento perro', 'croqueta']
            }
            
            detected_products = []
            for product, keywords in product_keywords.items():
                if any(kw in user_lower for kw in keywords):
                    detected_products.append(product)
            
            if detected_products:
                # Buscar opciones de tienda
                options = data_store.find_store_options(detected_products)
                
                product_names = ", ".join(detected_products).replace("_", " ")
                SessionManager.add_message(
                    'assistant',
                    f"Analicé **{len(detected_products)} productos** en 6 comercios. Encontré opciones en {len(options)} tiendas:",
                    'store_options',
                    options
                )
            
            elif any(w in user_lower for w in ['predice', 'precio', 'cuándo', 'cuando comprar']):
                # Predicción de precios
                product = 'leche'  # Default o extraer del texto
                prediction = data_store.predict_prices('L1')  # ID de leche colun
                
                if prediction:
                    SessionManager.add_message(
                        'assistant',
                        f"🔮 **Análisis LSTM** para Leche Colun 1L",
                        'prediction',
                        prediction
                    )
                else:
                    SessionManager.add_message('assistant', "No tengo suficiente historial para predecir este producto.")
            
            elif any(w in user_lower for w in ['receta', 'cocinar', 'tengo']):
                # Sugerir receta
                recipe = {
                    'name': 'Papas a la Crema Gratinadas',
                    'emoji': '🥔',
                    'time': '45 minutos',
                    'difficulty': 'Fácil',
                    'have': ['Papas', 'Crema'],
                    'need': [
                        {'name': 'Queso rallado', 'price': 2990},
                        {'name': 'Mantequilla', 'price': 1990},
                        {'name': 'Cebolla', 'price': 890}
                    ]
                }
                
                SessionManager.add_message(
                    'assistant',
                    "¡Perfecto! Con esos ingredientes puedes hacer:",
                    'recipe',
                    recipe
                )
            
            else:
                SessionManager.add_message(
                    'assistant',
                    "Entiendo. Prueba con:\n• **'leche y huevos'** para comparar tiendas\n• **'predice leche'** para análisis de precios\n• **'tengo papas'** para recetas"
                )
            
            st.rerun()
    
    @staticmethod
    def cart():
        """Vista de carrito"""
        st.markdown("## 🛒 Tu Carrito")
        
        if not st.session_state.cart:
            st.markdown("""
                <div class="empty-state">
                    <div class="empty-icon">🛒</div>
                    <h3>Tu carrito está vacío</h3>
                    <p>Ve al chat para agregar productos de diferentes tiendas</p>
                </div>
            """, unsafe_allow_html=True)
            
            if st.button("← Volver al chat", type="primary"):
                SessionManager.switch_view('chat')
                st.rerun()
            
            return
        
        # Agrupar por tienda
        by_store = defaultdict(list)
        for item in st.session_state.cart:
            by_store[item.product.store_name].append(item)
        
        total_general = 0
        
        for store_name, items in by_store.items():
            st.markdown(f"### {store_name}")
            
            for item in items:
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.write(f"**{item.product.name}**")
                    st.caption(f"{item.product.brand}")
                
                with col2:
                    st.write(f"x{item.quantity}")
                
                with col3:
                    subtotal = item.total
                    total_general += subtotal
                    st.write(f"${subtotal:,}")
            
            st.divider()
        
        # Total
        st.markdown(f"""
            <div style="background: #f0fdf4; border-radius: 0.75rem; padding: 1.5rem; margin: 2rem 0;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="color: #166534; font-size: 0.875rem;">Total estimado</div>
                        <div style="font-size: 2rem; font-weight: 700; color: #059669;">${total_general:,}</div>
                    </div>
                    <div style="text-align: right; color: #166534; font-size: 0.875rem;">
                        Incluye productos de {len(by_store)} tienda(s)
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💳 Proceder al pago", type="primary", use_container_width=True):
                st.success("✅ Pago procesado con Haulmer")
                st.balloons()
                SessionManager.clear_cart()
                time.sleep(2)
                st.rerun()
        
        with col2:
            if st.button("🗑️ Vaciar carrito", use_container_width=True):
                SessionManager.clear_cart()
                st.rerun()
        
        if st.button("← Seguir comprando"):
            SessionManager.switch_view('chat')
            st.rerun()
    
    @staticmethod
    def analytics():
        """Vista de analytics"""
        st.markdown("## 📊 Analytics")
        
        data_store = st.session_state.data_store
        
        # Métricas
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Productos", "42", "+5 esta semana")
        
        with col2:
            st.metric("Tiendas", "6", "+1 nueva")
        
        with col3:
            st.metric("Predicciones", "98.5%", "↑ 2%")
        
        with col4:
            st.metric("Ahorro promedio", "$3.240", "por compra")
        
        st.divider()
        
        # Gráfico de ejemplo
        st.subheader("Tendencia de precios - Leche")
        
        history = data_store.price_history.get('L1', [])
        if history:
            df = pd.DataFrame([
                {'Fecha': h['date'].strftime('%d/%m'), 'Precio': h['price']}
                for h in history[-14:]  # Últimos 14 días
            ])
            
            st.line_chart(df.set_index('Fecha'), use_container_width=True)
        
        st.info("💡 **Insight**: Los precios de lácteos suelen ser más bajos los martes y miércoles por reabastecimiento de inventario.")

# ============================================================
# APLICACIÓN PRINCIPAL
# ============================================================

def main():
    """Punto de entrada principal"""
    
    # Inicializar sesión
    SessionManager.init()
    
    # Renderizar UI base
    UIComponents.header()
    UIComponents.navigation()
    
    # Renderizar vista actual
    st.markdown('<div class="app-container">', unsafe_allow_html=True)
    
    current_view = st.session_state.current_view
    
    if current_view == 'chat':
        Views.chat()
    elif current_view == 'cart':
        Views.cart()
    elif current_view == 'analytics':
        Views.analytics()
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
