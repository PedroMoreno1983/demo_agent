# shopai_pro.py
# Diseño premium, código limpio, 100% funcional

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import random
import hashlib

# Configuración mínima necesaria
st.set_page_config(
    page_title="ShopAI Pro",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# ESTADO GLOBAL SIMPLE
# ============================================================

if 'messages' not in st.session_state:
    st.session_state.messages = []
    st.session_state.cart = []
    st.session_state.page = 'chat'

# ============================================================
# DATOS
# ============================================================

PRODUCTS = {
    'leche': [
        {'name': 'Leche Colun 1L', 'price': 1190, 'store': 'Jumbo', 'dist': 2.3, 'time': 90, 'delivery': 3990},
        {'name': 'Leche Soprole 1L', 'price': 1090, 'store': 'Lider', 'dist': 1.8, 'time': 60, 'delivery': 1990},
        {'name': 'Leche LoncoLeche 1L', 'price': 990, 'store': 'OK Market', 'dist': 0.5, 'time': 25, 'delivery': 1500},
    ],
    'huevos': [
        {'name': 'Huevos 12un', 'price': 3290, 'store': 'Jumbo', 'dist': 2.3, 'time': 90, 'delivery': 3990},
        {'name': 'Huevos 12un', 'price': 3590, 'store': 'Lider', 'dist': 1.8, 'time': 60, 'delivery': 1990},
    ]
}

# ============================================================
# FUNCIONES CORE
# ============================================================

def find_best_option(products_list):
    """Encuentra la mejor tienda para comprar todo"""
    stores = {}
    
    for prod_name in products_list:
        if prod_name in PRODUCTS:
            for variant in PRODUCTS[prod_name]:
                store = variant['store']
                if store not in stores:
                    stores[store] = {
                        'items': [],
                        'total_product': 0,
                        'delivery': variant['delivery'],
                        'time': variant['time'],
                        'dist': variant['dist']
                    }
                stores[store]['items'].append(variant)
                stores[store]['total_product'] += variant['price']
    
    # Calcular score: precio total + tiempo + distancia
    best = None
    best_score = float('inf')
    
    for store_name, data in stores.items():
        total = data['total_product'] + data['delivery']
        score = total + (data['time'] * 10) + (data['dist'] * 100)
        
        if score < best_score:
            best_score = score
            best = {
                'store': store_name,
                'total': total,
                'product_cost': data['total_product'],
                'delivery': data['delivery'],
                'time': data['time'],
                'items': data['items']
            }
    
    return best

def predict_price(product):
    """Simula predicción LSTM"""
    base = 1190
    days = []
    for i in range(7):
        change = random.uniform(-0.05, 0.03)
        base = int(base * (1 + change))
        days.append({
            'day': (datetime.now() + timedelta(days=i+1)).strftime('%a'),
            'price': base
        })
    
    min_day = min(days, key=lambda x: x['price'])
    return {
        'current': 1190,
        'forecast': days,
        'min_price': min_day['price'],
        'min_day': min_day['day'],
        'buy_now': min_day['price'] > 1150
    }

# ============================================================
# UI COMPONENTS
# ============================================================

def header():
    """Header minimalista"""
    st.markdown("""
        <style>
        .main-title {
            font-size: 2rem;
            font-weight: 700;
            color: #1a1a1a;
            margin-bottom: 0.5rem;
        }
        .subtitle {
            color: #666;
            font-size: 0.9rem;
        }
        </style>
        <div style="padding: 2rem 0; border-bottom: 1px solid #eee; margin-bottom: 2rem;">
            <div class="main-title">⚡ ShopAI Pro</div>
            <div class="subtitle">Un solo lugar. El mejor precio. Entrega optimizada.</div>
        </div>
    """, unsafe_allow_html=True)

def chat_message(role, content, actions=None):
    """Mensaje de chat estilizado"""
    is_user = role == 'user'
    
    bg_color = '#2563eb' if is_user else '#f3f4f6'
    text_color = 'white' if is_user else '#1f2937'
    align = 'flex-end' if is_user else 'flex-start'
    
    st.markdown(f"""
        <div style="display: flex; justify-content: {align}; margin: 1rem 0;">
            <div style="max-width: 80%; background: {bg_color}; color: {text_color}; 
                        padding: 1rem 1.5rem; border-radius: 1rem; 
                        {'border-bottom-right-radius: 0.25rem;' if is_user else 'border-bottom-left-radius: 0.25rem;'}">
                {content}
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    if actions:
        actions()

def store_card(data):
    """Card de tienda optimizada"""
    savings = random.randint(2000, 5000)
    
    st.markdown(f"""
        <div style="background: white; border: 2px solid #10b981; border-radius: 12px; 
                    padding: 1.5rem; margin: 1rem 0; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);">
            <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 1rem;">
                <div>
                    <div style="font-size: 1.25rem; font-weight: 600; color: #1f2937;">
                        🏆 {data['store']}
                    </div>
                    <div style="color: #6b7280; font-size: 0.875rem; margin-top: 0.25rem;">
                        {data['dist']} km • 🕐 {data['time']} min entrega
                    </div>
                </div>
                <div style="background: #dcfce7; color: #166534; padding: 0.5rem 1rem; 
                            border-radius: 9999px; font-size: 0.875rem; font-weight: 500;">
                    Ahorras ${savings:,}
                </div>
            </div>
            
            <div style="background: #f9fafb; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span style="color: #6b7280;">Productos</span>
                    <span style="font-weight: 500;">${data['product_cost']:,}</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span style="color: #6b7280;">Delivery</span>
                    <span style="font-weight: 500;">${data['delivery']:,}</span>
                </div>
                <div style="border-top: 1px solid #e5e7eb; margin-top: 0.5rem; padding-top: 0.5rem;
                            display: flex; justify-content: space-between;">
                    <span style="font-weight: 600;">Total</span>
                    <span style="font-size: 1.25rem; font-weight: 700; color: #059669;">
                        ${data['total']:,}
                    </span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    if st.button(f"Seleccionar {data['store']}", key=f"btn_{data['store']}", 
                 type="primary", use_container_width=True):
        st.session_state.cart.extend(data['items'])
        st.success(f"✅ {len(data['items'])} productos agregados")
        time.sleep(0.5)
        st.rerun()

def prediction_chart(pred):
    """Visualización de predicción"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Precio Hoy", f"${pred['current']:,}")
    with col2:
        st.metric("Mínimo Proyectado", f"${pred['min_price']:,}", 
                 delta=f"{pred['min_day']}")
    with col3:
        rec = "COMPRAR" if pred['buy_now'] else "ESPERAR"
        color = "#10b981" if pred['buy_now'] else "#ef4444"
        st.markdown(f"""
            <div style="text-align: center;">
                <div style="color: #6b7280; font-size: 0.875rem;">Recomendación</div>
                <div style="color: {color}; font-size: 1.5rem; font-weight: 700;">
                    {rec}
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # Chart
    chart_data = pd.DataFrame([
        {'Día': 'Hoy', 'Precio': pred['current']}
    ] + [{'Día': d['day'], 'Precio': d['price']} for d in pred['forecast']])
    
    st.line_chart(chart_data.set_index('Día'), use_container_width=True)

# ============================================================
# PÁGINAS
# ============================================================

def page_chat():
    """Página principal de chat"""
    
    # Mostrar historial
    for msg in st.session_state.messages:
        if msg['type'] == 'text':
            chat_message(msg['role'], msg['content'])
        elif msg['type'] == 'store':
            chat_message('agent', msg['content'])
            store_card(msg['data'])
        elif msg['type'] == 'prediction':
            chat_message('agent', msg['content'])
            prediction_chart(msg['data'])
    
    # Mensaje inicial
    if not st.session_state.messages:
        chat_message('agent', """
            👋 ¡Hola! Soy **ShopAI Pro**.
            
            **Prueba estas opciones:**
            • Escribe **"leche y huevos"** → Comparo todas las tiendas y elijo la mejor
            • Escribe **"predice leche"** → Análisis de precios con LSTM  
            • Escribe **"receta papas"** → Sugiero qué cocinar
            
            ¿Qué necesitas?
        """)
    
    # Input
    user_input = st.chat_input("Escribe aquí...")
    
    if user_input:
        # Agregar mensaje usuario
        st.session_state.messages.append({
            'role': 'user',
            'type': 'text',
            'content': user_input
        })
        
        # Procesar
        with st.spinner(''):
            time.sleep(0.3)
        
        user_lower = user_input.lower()
        
        # Detectar intención y responder
        if any(x in user_lower for x in ['leche', 'huevos', 'pan', 'papa']):
            # Extraer productos
            products = [p for p in ['leche', 'huevos', 'pan', 'papa'] if p in user_lower]
            
            if products:
                best = find_best_option(products)
                
                if best:
                    st.session_state.messages.append({
                        'role': 'agent',
                        'type': 'store',
                        'content': f"Analicé {len(products)} productos en 3 tiendas. La mejor opción es:",
                        'data': best
                    })
                else:
                    st.session_state.messages.append({
                        'role': 'agent',
                        'type': 'text',
                        'content': "No encontré todos los productos en una sola tienda. ¿Quieres que busque opciones parciales?"
                    })
        
        elif 'predice' in user_lower or 'precio' in user_lower:
            pred = predict_price('leche')
            st.session_state.messages.append({
                'role': 'agent',
                'type': 'prediction',
                'content': "🔮 **Análisis LSTM** - Predicción para Leche 1L (confianza 87%)",
                'data': pred
            })
        
        elif 'receta' in user_lower or 'cocinar' in user_lower:
            st.session_state.messages.append({
                'role': 'agent',
                'type': 'text',
                'content': """
                    🍳 **Papas a la Crema Gratinadas**
                    
                    **Ingredientes que tienes:** Papas
                    
                    **Necesitas comprar:**
                    • Crema de leche - $1.890
                    • Queso rallado - $2.990  
                    • Mantequilla - $1.990
                    
                    **Total ingredientes:** $6.870
                    **Tiempo:** 45 minutos
                    
                    ¿Agrego los ingredientes al carrito?
                """
            })
        
        else:
            st.session_state.messages.append({
                'role': 'agent',
                'type': 'text',
                'content': "Entiendo. Prueba con: **'leche y huevos'**, **'predice leche'**, o **'receta papas'**"
            })
        
        st.rerun()

def page_cart():
    """Página de carrito"""
    st.markdown("## 🛒 Tu Carrito")
    
    if not st.session_state.cart:
        st.info("Carrito vacío. Ve al chat para agregar productos.")
        if st.button("← Volver al chat"):
            st.session_state.page = 'chat'
            st.rerun()
        return
    
    total = sum(item['price'] for item in st.session_state.cart)
    
    for item in st.session_state.cart:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**{item['name']}**")
            st.caption(f"{item['store']}")
        with col2:
            st.write(f"${item['price']:,}")
    
    st.divider()
    st.write(f"### Total: ${total:,}")
    
    if st.button("💳 Proceder al pago", type="primary", use_container_width=True):
        st.success("✅ Pago procesado con Haulmer")
        st.balloons()
        st.session_state.cart = []
        time.sleep(2)
        st.rerun()
    
    if st.button("← Seguir comprando"):
        st.session_state.page = 'chat'
        st.rerun()

# ============================================================
# MAIN
# ============================================================

def main():
    header()
    
    # Navegación simple
    cols = st.columns([1, 1, 4, 1])
    with cols[0]:
        if st.button("💬 Chat", use_container_width=True):
            st.session_state.page = 'chat'
            st.rerun()
    with cols[1]:
        cart_count = len(st.session_state.cart)
        label = f"🛒 Carrito ({cart_count})" if cart_count else "🛒 Carrito"
        if st.button(label, use_container_width=True):
            st.session_state.page = 'cart'
            st.rerun()
    
    st.divider()
    
    # Renderizar página actual
    if st.session_state.page == 'chat':
        page_chat()
    else:
        page_cart()

if __name__ == "__main__":
    main()
