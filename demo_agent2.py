# shopai_working.py
# Versión estable sin dependencias problemáticas

import streamlit as st
import json
import random
import math
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional
import time
import hashlib
import base64
from collections import defaultdict

# Solo dependencias que vienen con Python o Streamlit
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="ShopAI - Agente de Compras",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CSS PROFESIONAL
# ============================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        margin: -4rem -4rem 2rem -4rem;
        text-align: center;
        color: white;
    }
    
    .brand-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    .brand-subtitle {
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    
    .chat-container {
        background: #f8fafc;
        border-radius: 20px;
        padding: 2rem;
        min-height: 400px;
        max-height: 500px;
        overflow-y: auto;
        border: 1px solid #e2e8f0;
    }
    
    .message {
        padding: 1rem 1.5rem;
        border-radius: 16px;
        margin-bottom: 1rem;
        max-width: 80%;
        animation: fadeIn 0.3s ease;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .message.user {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: auto;
        border-bottom-right-radius: 4px;
    }
    
    .message.agent {
        background: white;
        border: 1px solid #e2e8f0;
        border-bottom-left-radius: 4px;
    }
    
    .store-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 2px solid transparent;
        transition: all 0.3s;
    }
    
    .store-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px rgba(0,0,0,0.1);
    }
    
    .store-card.best {
        border-color: #10b981;
        background: #f0fdf4;
    }
    
    .price-tag {
        font-size: 1.5rem;
        font-weight: 700;
        color: #059669;
    }
    
    .savings {
        background: #dcfce7;
        color: #166534;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: 600;
    }
    
    .product-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
        gap: 1rem;
        margin-top: 1rem;
    }
    
    .product-card {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        transition: all 0.2s;
    }
    
    .product-card:hover {
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    
    .btn-primary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        cursor: pointer;
        width: 100%;
    }
    
    .recipe-card {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 2px solid #f59e0b;
    }
    
    .ingredient-have {
        background: #d1fae5;
        color: #065f46;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        margin: 0.25rem;
        font-size: 0.875rem;
    }
    
    .ingredient-need {
        background: #fee2e2;
        color: #991b1b;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        margin: 0.25rem;
        font-size: 0.875rem;
    }
    
    .prediction-box {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1f2937;
    }
    
    .metric-label {
        color: #6b7280;
        font-size: 0.875rem;
    }
    
    .vision-area {
        border: 3px dashed #667eea;
        border-radius: 16px;
        padding: 3rem;
        text-align: center;
        background: #f8fafc;
    }
    
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATOS DEL SISTEMA
# ============================================================

STORES = {
    "jumbo": {
        "name": "Jumbo",
        "icon": "🟢",
        "color": "#007932",
        "delivery": False,
        "delivery_fee": 3990,
        "distance": 2.3,
        "time": 90,
        "type": "Supermercado"
    },
    "lider": {
        "name": "Lider",
        "icon": "🔵",
        "color": "#0077C8",
        "delivery": True,
        "delivery_fee": 1990,
        "distance": 1.8,
        "time": 60,
        "type": "Supermercado"
    },
    "ok_market": {
        "name": "OK Market",
        "icon": "🏪",
        "color": "#FF6B35",
        "delivery": True,
        "delivery_fee": 1500,
        "distance": 0.5,
        "time": 25,
        "type": "Convenience"
    },
    "cruz_verde": {
        "name": "Cruz Verde",
        "icon": "💊",
        "color": "#00A650",
        "delivery": True,
        "delivery_fee": 1990,
        "distance": 1.2,
        "time": 35,
        "type": "Farmacia"
    },
    "salcobrand": {
        "name": "Salcobrand",
        "icon": "💊",
        "color": "#E31837",
        "delivery": True,
        "delivery_fee": 1500,
        "distance": 0.8,
        "time": 30,
        "type": "Farmacia"
    },
    "big_pet": {
        "name": "Big Pet",
        "icon": "🐾",
        "color": "#8B5CF6",
        "delivery": True,
        "delivery_fee": 2500,
        "distance": 4.2,
        "time": 45,
        "type": "Mascotas"
    }
}

PRODUCTS_DB = {
    "leche": [
        {"id": "L1", "name": "Leche Entera Colun 1L", "brand": "Colun", "price": 1190, "original": 1390, "store": "jumbo", "emoji": "🥛"},
        {"id": "L2", "name": "Leche Entera Soprole 1L", "brand": "Soprole", "price": 1090, "original": None, "store": "lider", "emoji": "🥛"},
        {"id": "L3", "name": "Leche LoncoLeche 1L", "brand": "LoncoLeche", "price": 990, "original": 1190, "store": "ok_market", "emoji": "🥛"},
    ],
    "huevos": [
        {"id": "H1", "name": "Huevos 12 unidades", "brand": "Campo Lindo", "price": 3290, "original": 3990, "store": "jumbo", "emoji": "🥚"},
        {"id": "H2", "name": "Huevos Color 12un", "brand": "Soprole", "price": 3590, "original": None, "store": "lider", "emoji": "🥚"},
        {"id": "H3", "name": "Huevos 6 unidades", "brand": "Local", "price": 1890, "original": None, "store": "ok_market", "emoji": "🥚"},
    ],
    "pan": [
        {"id": "P1", "name": "Pan Marraqueta 1kg", "brand": "Bimbo", "price": 1990, "original": None, "store": "jumbo", "emoji": "🍞"},
        {"id": "P2", "name": "Pan Molde 500g", "brand": "Ideal", "price": 1590, "original": 1890, "store": "lider", "emoji": "🍞"},
        {"id": "P3", "name": "Pan Baguette", "brand": "Local", "price": 1290, "original": None, "store": "ok_market", "emoji": "🥖"},
    ],
    "papa": [
        {"id": "PA1", "name": "Papas Lavadas 1kg", "brand": "Nature", "price": 1290, "original": None, "store": "jumbo", "emoji": "🥔"},
        {"id": "PA2", "name": "Papas Pre-cocidas 400g", "brand": "McCain", "price": 2490, "original": 2990, "store": "lider", "emoji": "🍟"},
    ],
    "crema": [
        {"id": "C1", "name": "Crema de Leche 200ml", "brand": "Colun", "price": 1890, "original": None, "store": "jumbo", "emoji": "🥛"},
        {"id": "C2", "name": "Crema para Batir 250ml", "brand": "Nestlé", "price": 2190, "original": 2490, "store": "lider", "emoji": "🥛"},
    ],
    "paracetamol": [
        {"id": "M1", "name": "Paracetamol 500mg 16comp", "brand": "Genérico", "price": 1990, "original": None, "store": "cruz_verde", "emoji": "💊"},
        {"id": "M2", "name": "Paracetamol 500mg 16comp", "brand": "Genérico", "price": 1890, "original": None, "store": "salcobrand", "emoji": "💊"},
    ],
    "alimento_perro": [
        {"id": "PE1", "name": "Alimento Perro Adulto 15kg", "brand": "Pro Plan", "price": 45990, "original": 52990, "store": "big_pet", "emoji": "🐕"},
        {"id": "PE2", "name": "Alimento Perro 10kg", "brand": "Royal Canin", "price": 38990, "original": None, "store": "big_pet", "emoji": "🐕"},
    ]
}

RECIPES = {
    "papas_crema": {
        "name": "Papas a la Crema Gratinadas",
        "emoji": "🥔",
        "time": "45 min",
        "difficulty": "Fácil",
        "have": ["Papas", "Crema"],
        "need": [
            {"name": "Queso rallado", "amount": "200g", "price": 2990},
            {"name": "Mantequilla", "amount": "50g", "price": 1990},
            {"name": "Cebolla", "amount": "1 unidad", "price": 890},
            {"name": "Ajo", "amount": "2 dientes", "price": 590},
            {"name": "Sal", "amount": "al gusto", "price": 490}
        ]
    },
    "omelette": {
        "name": "Omelette de Queso",
        "emoji": "🍳",
        "time": "15 min",
        "difficulty": "Muy fácil",
        "have": ["Huevos"],
        "need": [
            {"name": "Queso rallado", "amount": "50g", "price": 2990},
            {"name": "Mantequilla", "amount": "10g", "price": 1990},
            {"name": "Sal", "amount": "al gusto", "price": 490}
        ]
    }
}

# ============================================================
# FUNCIONES DEL AGENTE
# ============================================================

def find_best_store(items: List[str]) -> List[Dict]:
    """Encuentra el mejor comercio para una lista de items"""
    
    results = []
    
    for store_id, store_info in STORES.items():
        available_items = []
        total_product_price = 0
        
        for item in items:
            products = PRODUCTS_DB.get(item, [])
            # Buscar en este store
            match = next((p for p in products if p["store"] == store_id), None)
            if match:
                available_items.append(match)
                total_product_price += match["price"]
        
        if available_items:
            # Calcular costo total con delivery
            delivery = store_info["delivery_fee"] if not store_info["delivery"] else store_info["delivery_fee"]
            total = total_product_price + delivery
            
            # Score: combina precio, tiempo y distancia
            score = (total / 50000) * 0.5 + (store_info["time"] / 120) * 0.3 + (store_info["distance"] / 10) * 0.2
            
            results.append({
                "store_id": store_id,
                "store_info": store_info,
                "items": available_items,
                "product_total": total_product_price,
                "delivery_fee": delivery,
                "total": total,
                "time": store_info["time"],
                "score": score,
                "coverage": len(available_items) / len(items)
            })
    
    # Ordenar por score (menor es mejor)
    results.sort(key=lambda x: x["score"])
    return results

def predict_prices(product_name: str) -> Dict:
    """Simula predicción de precios con LSTM"""
    # Generar predicción determinista basada en hash
    h = hashlib.md5(product_name.encode()).hexdigest()
    random.seed(int(h[:8], 16))
    
    current = 1190
    predictions = []
    
    for day in range(7):
        change = random.uniform(-0.08, 0.05)
        pred_price = int(current * (1 + change))
        predictions.append({
            "day": (datetime.now() + timedelta(days=day+1)).strftime("%a %d"),
            "price": pred_price,
            "change": change
        })
        current = pred_price
    
    min_price = min(p["price"] for p in predictions)
    min_day = next(p["day"] for p in predictions if p["price"] == min_price)
    
    return {
        "current": 1190,
        "predictions": predictions,
        "min_price": min_price,
        "min_day": min_day,
        "recommendation": "wait" if min_price < 1150 else "buy_now",
        "confidence": random.uniform(0.75, 0.95)
    }

def detect_ingredients_image(image_bytes: bytes) -> List[Dict]:
    """Simula detección de ingredientes en imagen"""
    h = hashlib.md5(image_bytes).hexdigest()
    random.seed(int(h[:8], 16))
    
    ingredients = ["Papa", "Cebolla", "Huevo", "Leche", "Queso", "Zanahoria", "Pollo"]
    detected = random.sample(ingredients, random.randint(3, 5))
    
    return [{"name": ing, "confidence": random.uniform(0.75, 0.98)} for ing in detected]

def generate_weekly_menu(budget: int, family_size: int) -> Dict:
    """Genera menú semanal optimizado"""
    days = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
    meals = []
    total_cost = 0
    
    menu_templates = [
        {"name": "Pollo al horno con papas", "base_cost": 8500},
        {"name": "Pasta con salsa boloñesa", "base_cost": 6200},
        {"name": "Pescado a la plancha con arroz", "base_cost": 7800},
        {"name": "Lentejas con arroz", "base_cost": 4500},
        {"name": "Tortilla de papas", "base_cost": 3800},
        {"name": "Ensalada César con pollo", "base_cost": 5500},
        {"name": "Sopa de verduras", "base_cost": 3200}
    ]
    
    for i, day in enumerate(days):
        template = menu_templates[i % len(menu_templates)]
        cost = template["base_cost"] * family_size
        total_cost += cost
        
        meals.append({
            "day": day,
            "meal": template["name"],
            "cost": cost,
            "time": random.choice([30, 45, 60])
        })
    
    return {
        "meals": meals,
        "total": total_cost,
        "within_budget": total_cost <= budget,
        "savings": max(0, budget - total_cost)
    }

# ============================================================
# INTERFAZ PRINCIPAL
# ============================================================

def init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.cart = []
        st.session_state.view = "chat"

def render_chat():
    """Renderiza chat principal"""
    
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    if not st.session_state.messages:
        welcome = {
            "role": "agent",
            "content": """¡Hola! 👋 Soy **ShopAI**, tu asistente de compras inteligente.

**Puedo ayudarte con:**

🛒 **Comprar todo en un solo lugar** - Elijo el comercio más conveniente
🔮 **Predecir precios** - Te digo cuándo comprar para ahorrar
📸 **Detectar ingredientes** - Sube una foto y te sugiero recetas
📅 **Planificar la semana** - Menú completo optimizado por presupuesto

**¿Qué necesitas?** Escribe o selecciona una opción:"""
        }
        st.session_state.messages.append(welcome)
    
    for msg in st.session_state.messages:
        role_class = "user" if msg["role"] == "user" else "agent"
        st.markdown(f'<div class="message {role_class}">{msg["content"]}</div>', 
                   unsafe_allow_html=True)
        
        # Renderizar contenido especial
        if msg.get("stores"):
            render_store_comparison(msg["stores"])
        elif msg.get("recipe"):
            render_recipe(msg["recipe"])
        elif msg.get("prediction"):
            render_prediction(msg["prediction"])
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Input
    col1, col2 = st.columns([6, 1])
    with col1:
        user_input = st.text_input("", placeholder="Ej: 'Tengo papas y crema' o 'Predice precio de leche'", 
                                  label_visibility="collapsed", key="chat_input")
    with col2:
        if st.button("🎤", key="voice"):
            user_input = "Tengo papas y crema"
    
    if user_input:
        process_input(user_input)

def process_input(text: str):
    """Procesa entrada del usuario"""
    text_lower = text.lower()
    
    # Agregar mensaje usuario
    st.session_state.messages.append({"role": "user", "content": text})
    
    # Simular procesamiento
    with st.spinner("🧠 Pensando..."):
        time.sleep(0.5)
    
    # Detectar intención
    if any(w in text_lower for w in ["papa", "crema", "receta", "cocinar"]):
        # Buscar receta
        recipe = RECIPES["papas_crema"]
        response = {
            "role": "agent",
            "content": f"¡Perfecto! Con **papas y crema** puedes hacer:",
            "recipe": recipe
        }
    
    elif any(w in text_lower for w in ["predice", "precio", "va a subir", "cuándo comprar"]):
        # Predicción de precios
        pred = predict_prices("leche")
        response = {
            "role": "agent",
            "content": f"🔮 **Predicción para Leche Entera 1L** (Confianza: {pred['confidence']:.0%})",
            "prediction": pred
        }
    
    elif any(w in text_lower for w in ["menú", "semanal", "planificar"]):
        # Plan semanal
        response = {
            "role": "agent",
            "content": "📅 **Plan semanal generado** para 2 personas con $50.000"
        }
        # Aquí iría el render del plan
    
    elif any(w in text_lower for w in ["busco", "quiero", "necesito"]):
        # Búsqueda de productos con comparación de tiendas
        items = ["leche", "huevos"]  # Extraído del texto
        stores = find_best_store(items)
        
        best = stores[0] if stores else None
        if best:
            msg = f"Encontré **{len(best['items'])} productos**. La mejor opción es:"
            response = {
                "role": "agent",
                "content": msg,
                "stores": stores[:3]  # Top 3
            }
        else:
            response = {
                "role": "agent",
                "content": "No encontré todos los productos en un solo lugar. ¿Quieres que busque en múltiples tiendas?"
            }
    
    else:
        response = {
            "role": "agent",
            "content": """Entiendo. Prueba con:
• **"Tengo papas y crema"** - Sugiero recetas
• **"Predice precio de leche"** - Análisis LSTM  
• **"Menú semanal $50.000"** - Planificación
• **"Busco leche y huevos"** - Comparación de tiendas"""
        }
    
    st.session_state.messages.append(response)
    st.rerun()

def render_store_comparison(stores: List[Dict]):
    """Renderiza comparación de tiendas"""
    st.markdown("### 🏆 Opciones encontradas:")
    
    for idx, store in enumerate(stores):
        is_best = idx == 0
        card_class = "store-card best" if is_best else "store-card"
        badge = "⭐ MEJOR OPCIÓN" if is_best else f"#{idx+1}"
        
        savings = ""
        if is_best and len(stores) > 1:
            savings_diff = stores[1]["total"] - store["total"]
            if savings_diff > 0:
                savings = f'<span class="savings">Ahorras ${savings_diff:,}</span>'
        
        st.markdown(f"""
            <div class="{card_class}">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                    <div>
                        <h3>{store['store_info']['icon']} {store['store_info']['name']}</h3>
                        <p style="color: #6b7280; margin: 0;">{store['store_info']['type']} • {store['store_info']['distance']} km • 🕐 {store['time']} min</p>
                    </div>
                    <div style="text-align: right;">
                        <span style="background: {'#10b981' if is_best else '#e5e7eb'}; color: {'white' if is_best else '#374151'}; 
                             padding: 0.5rem 1rem; border-radius: 20px; font-weight: 600;">{badge}</span>
                        {savings}
                    </div>
                </div>
                
                <div style="display: grid; grid-template-columns: 2fr 1fr 1fr; gap: 1rem; margin: 1rem 0; padding: 1rem; background: #f9fafb; border-radius: 8px;">
                    <div>
                        <p style="margin: 0; color: #6b7280; font-size: 0.875rem;">Productos</p>
                        <p style="margin: 0; font-weight: 600;">{', '.join(item['name'][:20] for item in store['items'])}</p>
                    </div>
                    <div>
                        <p style="margin: 0; color: #6b7280; font-size: 0.875rem;">Subtotal</p>
                        <p style="margin: 0; font-weight: 600;">${store['product_total']:,}</p>
                    </div>
                    <div>
                        <p style="margin: 0; color: #6b7280; font-size: 0.875rem;">Delivery</p>
                        <p style="margin: 0; font-weight: 600;">${store['delivery_fee']:,}</p>
                    </div>
                </div>
                
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span class="price-tag">${store['total']:,}</span>
                        <span style="color: #6b7280; margin-left: 0.5rem;">total</span>
                    </div>
                    <button class="btn-primary" style="width: auto; padding: 0.5rem 2rem;">Seleccionar</button>
                </div>
            </div>
        """, unsafe_allow_html=True)

def render_recipe(recipe: Dict):
    """Renderiza receta"""
    total_missing = sum(ing["price"] for ing in recipe["need"])
    
    st.markdown(f"""
        <div class="recipe-card">
            <h3>{recipe['emoji']} {recipe['name']}</h3>
            <p>⏱️ {recipe['time']} • {recipe['difficulty']}</p>
            
            <div style="margin: 1rem 0;">
                <p style="font-weight: 600; margin-bottom: 0.5rem;">✅ Ingredientes que tienes:</p>
                {''.join(f'<span class="ingredient-have">{ing}</span>' for ing in recipe['have'])}
            </div>
            
            <div style="margin: 1rem 0;">
                <p style="font-weight: 600; margin-bottom: 0.5rem;">🛒 Ingredientes a comprar (${total_missing:,}):</p>
                {''.join(f'<span class="ingredient-need' + str(ing["price"]) + '</span>' for ing in recipe['need'])}
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    if st.button("🛒 Agregar ingredientes al carrito", type="primary"):
        st.success("✅ Agregados!")

def render_prediction(pred: Dict):
    """Renderiza predicción de precios"""
    recommendation = "🟢 COMPRAR AHORA" if pred["recommendation"] == "buy_now" else "🔴 ESPERAR"
    color = "#10b981" if pred["recommendation"] == "buy_now" else "#ef4444"
    
    st.markdown(f"""
        <div class="prediction-box">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                <div>
                    <p style="margin: 0; color: #6b7280;">Precio actual</p>
                    <p style="margin: 0; font-size: 1.5rem; font-weight: 700;">${pred['current']:,}</p>
                </div>
                <div style="text-align: center;">
                    <p style="margin: 0; color: #6b7280;">Mejor día para comprar</p>
                    <p style="margin: 0; font-size: 1.25rem; font-weight: 600;">{pred['min_day']}</p>
                    <p style="margin: 0; color: #059669; font-weight: 700;">${pred['min_price']:,}</p>
                </div>
                <div style="text-align: right;">
                    <p style="margin: 0; color: #6b7280;">Recomendación</p>
                    <p style="margin: 0; font-size: 1.25rem; font-weight: 700; color: {color};">{recommendation}</p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Gráfico simple de predicción
    chart_data = pd.DataFrame([
        {"Día": "Hoy", "Precio": pred["current"]}
    ] + [{"Día": p["day"], "Precio": p["price"]} for p in pred["predictions"]])
    
    st.line_chart(chart_data.set_index("Día"))

def render_sidebar():
    """Sidebar con carrito"""
    with st.sidebar:
        st.header("🛒 Tu Compra")
        
        if st.session_state.cart:
            total = sum(item["price"] for item in st.session_state.cart)
            st.write(f"**Total: ${total:,}**")
            
            if st.button("💳 Pagar", type="primary", use_container_width=True):
                st.success("Procesando...")
        else:
            st.info("Carrito vacío")
        
        st.divider()
        
        # Navegación rápida
        st.subheader("⚡ Accesos Rápidos")
        if st.button("🔮 Predicción de Precios"):
            st.session_state.messages.append({
                "role": "user",
                "content": "Predice precio de leche"
            })
            process_input("Predice precio de leche")
        
        if st.button("📅 Plan Semanal"):
            st.info("Menú generado (simulado)")
        
        if st.button("📸 Escanear Nevera"):
            st.info("Abriendo cámara... (simulado)")

def main():
    init_session()
    
    # Header
    st.markdown("""
        <div class="main-header">
            <h1 class="brand-title">🛍️ ShopAI</h1>
            <p class="brand-subtitle">Predicción LSTM • Visión por Computadora • Planificación Inteligente</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        render_chat()
    
    with col2:
        render_sidebar()

if __name__ == "__main__":
    main()
