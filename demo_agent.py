# demo_agent.py
# Guarda este archivo y ejecútalo con: streamlit run demo_agent.py

import streamlit as st
import json
import random
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional
import time

# Configuración de página
st.set_page_config(
    page_title="🛒 Smart Shopping Agent - Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado profesional
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    .chat-container {
        background: #f9fafb;
        border-radius: 20px;
        padding: 2rem;
        min-height: 400px;
        max-height: 600px;
        overflow-y: auto;
        border: 1px solid #e5e7eb;
    }
    
    .message-user {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 5px 20px;
        margin: 1rem 0 1rem 20%;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        animation: slideIn 0.3s ease-out;
    }
    
    .message-agent {
        background: white;
        color: #1f2937;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 20px 5px;
        margin: 1rem 20% 1rem 0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border: 1px solid #e5e7eb;
        animation: slideIn 0.3s ease-out;
    }
    
    .message-time {
        font-size: 0.75rem;
        opacity: 0.7;
        margin-top: 0.5rem;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .product-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 0.5rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        border: 1px solid #e5e7eb;
        transition: all 0.3s ease;
    }
    
    .product-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
    }
    
    .price-tag {
        font-size: 1.5rem;
        font-weight: 700;
        color: #059669;
    }
    
    .original-price {
        text-decoration: line-through;
        color: #9ca3af;
        font-size: 1rem;
    }
    
    .discount-badge {
        background: #dc2626;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        font-weight: 600;
    }
    
    .supermarket-tag {
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.75rem;
        background: #dbeafe;
        color: #1e40af;
        border-radius: 9999px;
        font-size: 0.875rem;
        font-weight: 500;
        margin-top: 0.5rem;
    }
    
    .recipe-card {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        border: 2px solid #f59e0b;
    }
    
    .ingredient-have {
        background: #d1fae5;
        color: #065f46;
        padding: 0.5rem 1rem;
        border-radius: 9999px;
        display: inline-block;
        margin: 0.25rem;
        font-size: 0.875rem;
    }
    
    .ingredient-need {
        background: #fee2e2;
        color: #991b1b;
        padding: 0.5rem 1rem;
        border-radius: 9999px;
        display: inline-block;
        margin: 0.25rem;
        font-size: 0.875rem;
    }
    
    .cart-item {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid #e5e7eb;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .btn-primary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 12px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s;
    }
    
    .btn-primary:hover {
        transform: scale(1.05);
        box-shadow: 0 10px 20px -5px rgba(102, 126, 234, 0.4);
    }
    
    .status-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.5rem 1rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        font-weight: 600;
    }
    
    .status-processing {
        background: #dbeafe;
        color: #1e40af;
    }
    
    .status-delivered {
        background: #d1fae5;
        color: #065f46;
    }
    
    .typing-indicator {
        display: flex;
        gap: 0.5rem;
        padding: 1rem;
    }
    
    .typing-dot {
        width: 8px;
        height: 8px;
        background: #9ca3af;
        border-radius: 50%;
        animation: typing 1.4s infinite;
    }
    
    .typing-dot:nth-child(2) { animation-delay: 0.2s; }
    .typing-dot:nth-child(3) { animation-delay: 0.4s; }
    
    @keyframes typing {
        0%, 60%, 100% { transform: translateY(0); }
        30% { transform: translateY(-10px); }
    }
    
    .voice-btn {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border: none;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        color: white;
        font-size: 1.5rem;
        cursor: pointer;
        transition: all 0.3s;
    }
    
    .voice-btn:hover {
        transform: scale(1.1);
        box-shadow: 0 10px 20px -5px rgba(245, 87, 108, 0.4);
    }
    
    .voice-btn.recording {
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(245, 87, 108, 0.7); }
        70% { box-shadow: 0 0 0 20px rgba(245, 87, 108, 0); }
        100% { box-shadow: 0 0 0 0 rgba(245, 87, 108, 0); }
    }
</style>
""", unsafe_allow_html=True)

# Datos de demo
SUPERMARKETS = {
    "jumbo": {"name": "🟢 Jumbo", "color": "#007932", "delivery": False},
    "lider": {"name": "🔵 Lider", "color": "#0077C8", "delivery": True},
    "santa_isabel": {"name": "🟡 Santa Isabel", "color": "#FFD700", "delivery": False},
    "unimarc": {"name": "🟠 Unimarc", "color": "#FF6B00", "delivery": True}
}

PRODUCTS_DB = {
    "leche": [
        {"id": 1, "name": "Leche Entera Colun 1L", "brand": "Colun", "price": 1190, "original_price": 1390, "supermarket": "jumbo", "image": "🥛", "category": "Lácteos"},
        {"id": 2, "name": "Leche Entera Soprole 1L", "brand": "Soprole", "price": 1090, "original_price": None, "supermarket": "lider", "image": "🥛", "category": "Lácteos"},
        {"id": 3, "name": "Leche Descremada LoncoLeche 1L", "brand": "LoncoLeche", "price": 990, "original_price": 1190, "supermarket": "santa_isabel", "image": "🥛", "category": "Lácteos"},
    ],
    "huevos": [
        {"id": 4, "name": "Huevos Blancos 12 unidades", "brand": "Campo Lindo", "price": 3290, "original_price": 3990, "supermarket": "jumbo", "image": "🥚", "category": "Huevos"},
        {"id": 5, "name": "Huevos Color 12 unidades", "brand": "Soprole", "price": 3590, "original_price": None, "supermarket": "lider", "image": "🥚", "category": "Huevos"},
    ],
    "pan": [
        {"id": 6, "name": "Pan Marraqueta 1kg", "brand": "Bimbo", "price": 1990, "original_price": None, "supermarket": "jumbo", "image": "🍞", "category": "Panadería"},
        {"id": 7, "name": "Pan Molde Blanco 500g", "brand": "Ideal", "price": 1590, "original_price": 1890, "supermarket": "unimarc", "image": "🍞", "category": "Panadería"},
    ],
    "papa": [
        {"id": 8, "name": "Papas Lavadas 1kg", "brand": "Nature", "price": 1290, "original_price": None, "supermarket": "jumbo", "image": "🥔", "category": "Verduras"},
        {"id": 9, "name": "Papas Deshidratadas 500g", "brand": "McCain", "price": 2490, "original_price": 2990, "supermarket": "lider", "image": "🥔", "category": "Congelados"},
    ],
    "crema": [
        {"id": 10, "name": "Crema de Leche 200ml", "brand": "Colun", "price": 1890, "original_price": None, "supermarket": "jumbo", "image": "🥛", "category": "Lácteos"},
        {"id": 11, "name": "Crema para Batir 250ml", "brand": "Nestlé", "price": 2190, "original_price": 2490, "supermarket": "santa_isabel", "image": "🥛", "category": "Lácteos"},
    ]
}

RECIPES_DB = {
    "papas_crema": {
        "name": "Papas a la Crema Gratinadas",
        "time": "45 min",
        "difficulty": "Fácil ⭐⭐",
        "servings": 4,
        "image": "🥔",
        "ingredients_have": ["Papas", "Crema"],
        "ingredients_need": [
            {"name": "Queso rallado", "amount": "200g", "price": 2990, "supermarket": "jumbo"},
            {"name": "Mantequilla", "amount": "50g", "price": 1990, "supermarket": "jumbo"},
            {"name": "Cebolla", "amount": "1 unidad", "price": 890, "supermarket": "lider"},
            {"name": "Ajo", "amount": "2 dientes", "price": 590, "supermarket": "lider"},
            {"name": "Sal y pimienta", "amount": "al gusto", "price": 1290, "supermarket": "unimarc"}
        ],
        "instructions": [
            "Precalienta el horno a 180°C",
            "Pela y corta las papas en rodajas finas",
            "Sofríe la cebolla y ajo en mantequilla",
            "Mezcla las papas con la crema y la mitad del queso",
            "Hornea por 30 minutos y agrega el queso restante al final"
        ]
    },
    "leche_huevos": {
        "name": "Torta de Leche y Huevos",
        "time": "60 min",
        "difficulty": "Medio ⭐⭐⭐",
        "servings": 6,
        "image": "🍰",
        "ingredients_have": ["Leche", "Huevos"],
        "ingredients_need": [
            {"name": "Harina", "amount": "250g", "price": 1590, "supermarket": "jumbo"},
            {"name": "Azúcar", "amount": "200g", "price": 1290, "supermarket": "lider"},
            {"name": "Mantequilla", "amount": "100g", "price": 1990, "supermarket": "jumbo"},
            {"name": "Esencia de vainilla", "amount": "1 cda", "price": 2490, "supermarket": "santa_isabel"}
        ],
        "instructions": [
            "Bate los huevos con el azúcar hasta blanquear",
            "Agrega la leche y la vainilla",
            "Incorpora la harina tamizada",
            "Hornea a 170°C por 45 minutos"
        ]
    }
}

# Inicializar estado de sesión
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.cart = []
    st.session_state.current_view = "chat"
    st.session_state.checkout_step = 1
    st.session_state.order_confirmed = False
    st.session_state.is_recording = False
    st.session_state.user_preferences = {
        "supermarket": None,
        "budget": None,
        "dietary": []
    }

def get_time():
    return datetime.now().strftime("%H:%M")

def add_message(role, content, message_type="text", data=None):
    st.session_state.messages.append({
        "role": role,
        "content": content,
        "type": message_type,
        "data": data,
        "time": get_time()
    })

def simulate_typing():
    """Simula indicador de escritura"""
    placeholder = st.empty()
    placeholder.markdown("""
        <div class="message-agent">
            <div class="typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    time.sleep(1.5)
    placeholder.empty()

def find_best_prices(query):
    """Busca mejores precios en todos los supermercados"""
    results = []
    query_lower = query.lower()
    
    for keyword, products in PRODUCTS_DB.items():
        if keyword in query_lower or any(word in query_lower for word in keyword.split()):
            results.extend(products)
    
    # Si no hay match exacto, buscar similitud simple
    if not results:
        for products in PRODUCTS_DB.values():
            for p in products:
                if any(word in p["name"].lower() for word in query_lower.split()):
                    results.append(p)
    
    return results[:6]  # Limitar resultados

def calculate_savings(products):
    """Calcula ahorros entre supermercados"""
    if len(products) < 2:
        return []
    
    # Agrupar por categoría similar
    best_deals = []
    for p1 in products:
        for p2 in products:
            if p1["id"] != p2["id"] and p1["category"] == p2["category"]:
                if p1["price"] < p2["price"]:
                    savings = p2["price"] - p1["price"]
                    best_deals.append({
                        "product": p1,
                        "alternative": p2,
                        "savings": savings,
                        "savings_percent": (savings / p2["price"]) * 100
                    })
    
    return sorted(best_deals, key=lambda x: x["savings"], reverse=True)[:3]

def process_user_input(text):
    """Procesa entrada del usuario y genera respuesta"""
    
    text_lower = text.lower()
    
    # Detectar intención
    if any(word in text_lower for word in ["hola", "buenas", "hey", "hi"]):
        return {
            "type": "welcome",
            "message": """¡Hola! 👋 Soy tu **Smart Shopping Agent**.

Puedo ayudarte a:

🛒 **Comprar productos** al mejor precio
🍳 **Sugerir recetas** con lo que tienes en casa
📊 **Comparar precios** entre supermercados
🚚 **Coordinar delivery** automático

**¿Qué necesitas hoy?** Puedes escribirme o usar el botón de micrófono 🎤"""
        }
    
    elif any(word in text_lower for word in ["receta", "cocinar", "preparar", "hacer con"]):
        # Detectar ingredientes
        ingredients_found = []
        for ingredient in ["papa", "crema", "leche", "huevo", "pan", "arroz", "pollo"]:
            if ingredient in text_lower:
                ingredients_found.append(ingredient)
        
        if "papa" in ingredients_found and "crema" in ingredients_found:
            return {
                "type": "recipe",
                "message": "¡Excelente combinación! 🥔✨ Con **papas y crema** puedo sugerirte:",
                "recipe": RECIPES_DB["papas_crema"]
            }
        elif "leche" in ingredients_found and "huevo" in ingredients_found:
            return {
                "type": "recipe",
                "message": "¡Perfecto! 🥛🥚 Con **leche y huevos** puedes hacer:",
                "recipe": RECIPES_DB["leche_huevos"]
            }
        else:
            return {
                "type": "text",
                "message": f"Veo que mencionaste: {', '.join(ingredients_found)}. ¿Qué otros ingredientes tienes? Así puedo sugerirte la mejor receta. 🍳"
            }
    
    elif any(word in text_lower for word in ["buscar", "quiero", "necesito", "precio", "comprar"]):
        products = find_best_prices(text)
        
        if products:
            best_deals = calculate_savings(products)
            return {
                "type": "products",
                "message": f"Encontré **{len(products)} productos** para ti. Aquí están las mejores opciones:",
                "products": products,
                "best_deals": best_deals
            }
        else:
            return {
                "type": "text",
                "message": "No encontré productos con ese nombre. ¿Puedes ser más específico? Por ejemplo: 'leche', 'huevos', 'pan', etc. 🤔"
            }
    
    elif any(word in text_lower for word in ["comparar", "diferencia", "más barato"]):
        products = find_best_prices(text.replace("comparar", "").replace("más barato", ""))
        best_deals = calculate_savings(products)
        
        if best_deals:
            return {
                "type": "comparison",
                "message": "📊 **Análisis de precios:** Aquí puedes ahorrar:",
                "best_deals": best_deals
            }
        else:
            return {
                "type": "text",
                "message": "Necesito saber qué producto quieres comparar. ¿Qué estás buscando? 🔍"
            }
    
    elif any(word in text_lower for word in ["carrito", "ver carrito", "mi compra"]):
        if st.session_state.cart:
            total = sum(item["price"] * item["quantity"] for item in st.session_state.cart)
            return {
                "type": "cart",
                "message": f"🛒 **Tu carrito** ({len(st.session_state.cart)} items) - Total: **${total:,.0f}**",
                "show_checkout": True
            }
        else:
            return {
                "type": "text",
                "message": "Tu carrito está vacío. ¡Agrega algunos productos! 🛍️"
            }
    
    elif any(word in text_lower for word in ["pagar", "checkout", "finalizar"]):
        if st.session_state.cart:
            return {
                "type": "checkout",
                "message": "💳 **Proceso de pago iniciado**",
                "step": 1
            }
        else:
            return {
                "type": "text",
                "message": "Primero agrega productos a tu carrito antes de pagar. 🛒"
            }
    
    elif any(word in text_lower for word in ["gracias", "chao", "adiós", "hasta luego"]):
        return {
            "type": "text",
            "message": "¡De nada! 😊 Estoy aquí cuando me necesites. ¡Que tengas un excelente día! 👋"
        }
    
    else:
        return {
            "type": "text",
            "message": """Entiendo. Puedo ayudarte con:

• **Buscar productos**: "Quiero leche"
• **Sugerir recetas**: "Tengo papas y crema"
• **Comparar precios**: "¿Dónde está más barato?"
• **Ver carrito**: "Ver mi carrito"

¿Qué necesitas? 🤔"""
        }

def render_chat():
    """Renderiza el área de chat"""
    
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Mensaje inicial si está vacío
    if not st.session_state.messages:
        welcome_msg = {
            "role": "agent",
            "content": """¡Hola! 👋 Soy tu **Smart Shopping Agent** 🤖

Estoy aquí para ayudarte a:

🛒 **Comprar** productos al mejor precio  
🍳 **Cocinar** con lo que tienes en casa  
📊 **Comparar** precios entre supermercados  
🚚 **Coordinar** delivery automático

**¿Qué necesitas hoy?**""",
            "type": "text",
            "time": get_time()
        }
        st.session_state.messages.append(welcome_msg)
    
    # Renderizar mensajes
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""
                <div class="message-user">
                    {msg["content"]}
                    <div class="message-time">{msg["time"]}</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="message-agent">
                    {msg["content"]}
                    <div class="message-time">{msg["time"]}</div>
                </div>
            """, unsafe_allow_html=True)
            
            # Renderizar contenido especial según tipo
            if msg.get("type") == "products" and msg.get("data"):
                render_products(msg["data"]["products"], msg["data"].get("best_deals"))
            elif msg.get("type") == "recipe" and msg.get("data"):
                render_recipe(msg["data"])
            elif msg.get("type") == "comparison" and msg.get("data"):
                render_comparison(msg["data"])
            elif msg.get("type") == "cart":
                render_cart()
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_products(products, best_deals=None):
    """Renderiza tarjetas de productos"""
    
    cols = st.columns(3)
    for idx, product in enumerate(products):
        with cols[idx % 3]:
            savings = ""
            if product.get("original_price"):
                discount = int(((product["original_price"] - product["price"]) / product["original_price"]) * 100)
                savings = f'<span class="discount-badge">-{discount}%</span>'
            
            st.markdown(f"""
                <div class="product-card">
                    <div style="font-size: 4rem; text-align: center;">{product["image"]}</div>
                    <h4>{product["name"]}</h4>
                    <p style="color: #6b7280; font-size: 0.9rem;">{product["brand"]}</p>
                    <div style="margin: 1rem 0;">
                        <span class="price-tag">${product["price"]:,.0f}</span>
                        {"<span class='original-price'>${:,.0f}</span>".format(product["original_price"]) if product.get("original_price") else ""}
                        {savings}
                    </div>
                    <span class="supermarket-tag">{SUPERMARKETS[product["supermarket"]]["name"]}</span>
                </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns([3, 1])
            with col1:
                qty = st.number_input(f"Cantidad", min_value=1, max_value=10, value=1, key=f"qty_{product['id']}", label_visibility="collapsed")
            with col2:
                if st.button("➕", key=f"add_{product['id']}", help="Agregar al carrito"):
                    add_to_cart(product, qty)
                    st.success("✅ Agregado!")
                    time.sleep(0.5)
                    st.rerun()
    
    # Mostrar mejores ofertas si existen
    if best_deals:
        st.markdown("---")
        st.subheader("💰 **Mejores Oportunidades de Ahorro**")
        for deal in best_deals:
            st.info(f"""
                **Ahorra ${deal['savings']:,.0f}** ({deal['savings_percent']:.1f}%) 
                comprando **{deal['product']['name']}** en {SUPERMARKETS[deal['product']['supermarket']]['name']}
                en vez de {SUPERMARKETS[deal['alternative']['supermarket']]['name']}
            """)

def render_recipe(recipe_data):
    """Renderiza tarjeta de receta"""
    
    recipe = recipe_data["recipe"]
    
    st.markdown(f"""
        <div class="recipe-card">
            <h2>{recipe["image"]} {recipe["name"]}</h2>
            <p>⏱️ {recipe["time"]} | {recipe["difficulty"]} | 🍽️ {recipe["servings"]} porciones</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**✅ Ingredientes que tienes:**")
        for ing in recipe["ingredients_have"]:
            st.markdown(f'<span class="ingredient-have">{ing}</span>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("**🛒 Ingredientes a comprar:**")
        total_need = 0
        for ing in recipe["ingredients_need"]:
            st.markdown(f'<span class="ingredient-need">{ing["name"]} ({ing["amount"]}) - ${ing["price"]:,.0f}</span>', unsafe_allow_html=True)
            total_need += ing["price"]
        
        st.markdown(f"**Total a comprar: ${total_need:,.0f}**")
    
    with st.expander("👨‍🍳 Ver instrucciones"):
        for i, step in enumerate(recipe["instructions"], 1):
            st.write(f"**{i}.** {step}")
    
    if st.button("🛒 Agregar ingredientes al carrito", type="primary", key="add_recipe_items"):
        for ing in recipe["ingredients_need"]:
            add_to_cart({
                "name": ing["name"],
                "price": ing["price"],
                "supermarket": ing["supermarket"],
                "image": "🛒",
                "id": random.randint(1000, 9999)
            }, 1)
        st.success(f"✅ {len(recipe['ingredients_need'])} ingredientes agregados!")
        time.sleep(1)
        st.rerun()

def render_comparison(comparison_data):
    """Renderiza comparación de precios"""
    
    best_deals = comparison_data["best_deals"]
    
    for deal in best_deals:
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            st.markdown(f"""
                <div style="background: white; padding: 1rem; border-radius: 10px; border: 2px solid #10b981;">
                    <h4>✅ {deal['product']['name']}</h4>
                    <p class="price-tag">${deal['product']['price']:,.0f}</p>
                    <span class="supermarket-tag">{SUPERMARKETS[deal['product']['supermarket']]['name']}</span>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div style="background: #fee2e2; padding: 1rem; border-radius: 10px; border: 1px solid #ef4444;">
                    <h4>❌ {deal['alternative']['name']}</h4>
                    <p style="text-decoration: line-through; color: #6b7280;">${deal['alternative']['price']:,.0f}</p>
                    <span class="supermarket-tag">{SUPERMARKETS[deal['alternative']['supermarket']]['name']}</span>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
                <div style="text-align: center; padding: 1rem;">
                    <h3 style="color: #059669;">💰 Ahorro</h3>
                    <h2>${deal['savings']:,.0f}</h2>
                    <p>({deal['savings_percent']:.1f}%)</p>
                </div>
            """, unsafe_allow_html=True)
        
        st.divider()

def render_cart():
    """Renderiza carrito de compras"""
    
    if not st.session_state.cart:
        st.info("Tu carrito está vacío")
        return
    
    total = 0
    for item in st.session_state.cart:
        item_total = item["price"] * item["quantity"]
        total += item_total
        
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
        with col1:
            st.write(f"**{item['name']}**")
            st.caption(f"{SUPERMARKETS[item['supermarket']]['name']}")
        with col2:
            st.write(f"${item['price']:,.0f}")
        with col3:
            st.write(f"x{item['quantity']}")
        with col4:
            st.write(f"**${item_total:,.0f}**")
    
    st.divider()
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("**Total del carrito:**")
    with col2:
        st.markdown(f"<h3 style='color: #059669;'>${total:,.0f}</h3>", unsafe_allow_html=True)
    
    # Detectar si necesita delivery de terceros
    needs_external_delivery = any(
        not SUPERMARKETS[item["supermarket"]]["delivery"] 
        for item in st.session_state.cart
    )
    
    if needs_external_delivery:
        st.warning("⚠️ Algunos productos requieren delivery externo (Uber Direct): +$3,990")
        total += 3990
        st.info(f"**Total con delivery: ${total:,.0f}**")
    
    if st.button("💳 Proceder al Pago", type="primary", use_container_width=True):
        st.session_state.current_view = "checkout"
        st.rerun()

def add_to_cart(product, quantity):
    """Agrega producto al carrito"""
    
    existing = next((item for item in st.session_state.cart if item["name"] == product["name"]), None)
    
    if existing:
        existing["quantity"] += quantity
    else:
        st.session_state.cart.append({
            "id": product["id"],
            "name": product["name"],
            "price": product["price"],
            "supermarket": product["supermarket"],
            "quantity": quantity,
            "image": product.get("image", "🛒")
        })

def render_checkout():
    """Renderiza proceso de checkout con Haulmer"""
    
    st.markdown('<h1 class="main-header">💳 Checkout Seguro</h1>', unsafe_allow_html=True)
    
    total = sum(item["price"] * item["quantity"] for item in st.session_state.cart)
    
    # Verificar si necesita delivery externo
    needs_uber = any(not SUPERMARKETS[item["supermarket"]]["delivery"] for item in st.session_state.cart)
    
    if needs_uber:
        delivery_fee = 3990
        st.info("🚚 Se agregó delivery Uber Direct (supermercado sin servicio propio)")
    else:
        delivery_fee = 0
        st.success("✅ Delivery incluido por el supermercado")
    
    final_total = total + delivery_fee
    
    # Paso 1: Resumen
    if st.session_state.checkout_step == 1:
        st.subheader("1️⃣ Resumen de tu compra")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            for item in st.session_state.cart:
                st.write(f"• {item['name']} x{item['quantity']} = ${item['price']*item['quantity']:,.0f}")
            
            if delivery_fee > 0:
                st.write(f"• Delivery Uber Direct = ${delivery_fee:,.0f}")
        
        with col2:
            st.markdown(f"""
                <div style="background: #f3f4f6; padding: 1.5rem; border-radius: 15px;">
                    <h4>Total a pagar</h4>
                    <h2 style="color: #059669;">${final_total:,.0f}</h2>
                    <p style="font-size: 0.8rem; color: #6b7280;">Incluye IVA</p>
                </div>
            """, unsafe_allow_html=True)
        
        if st.button("Continuar →", type="primary"):
            st.session_state.checkout_step = 2
            st.rerun()
    
    # Paso 2: Datos personales
    elif st.session_state.checkout_step == 2:
        st.subheader("2️⃣ Datos de facturación")
        
        with st.form("checkout_form"):
            col1, col2 = st.columns(2)
            with col1:
                nombre = st.text_input("Nombre completo*", value="Juan Pérez")
                email = st.text_input("Email*", value="juan@email.com")
                telefono = st.text_input("Teléfono*", value="+56912345678")
            with col2:
                rut = st.text_input("RUT*", value="12.345.678-9")
                direccion = st.text_input("Dirección de entrega*", value="Av. Las Condes 1234, Depto 501")
                comuna = st.selectbox("Comuna*", ["Las Condes", "Vitacura", "La Reina", "Ñuñoa", "Providencia"])
            
            tipo_doc = st.radio("Tipo de documento", ["Boleta electrónica (39)", "Factura electrónica (33)"])
            
            st.markdown("**Método de pago:**")
            st.image("https://via.placeholder.com/400x100/667eea/ffffff?text=Haulmer+Payment+Secure", use_column_width=True)
            st.caption("🔒 Pago seguro procesado por Haulmer TUU")
            
            submitted = st.form_submit_button("Pagar ahora", type="primary")
            
            if submitted:
                with st.spinner("Procesando pago con Haulmer..."):
                    time.sleep(2)
                    st.session_state.checkout_step = 3
                    st.session_state.payment_id = f"PAY-{random.randint(100000, 999999)}"
                    st.rerun()
    
    # Paso 3: Procesando pago
    elif st.session_state.checkout_step == 3:
        st.subheader("3️⃣ Procesando pago...")
        
        progress_bar = st.progress(0)
        
        for i in range(100):
            time.sleep(0.03)
            progress_bar.progress(i + 1)
        
        # Simular resultado exitoso
        st.session_state.checkout_step = 4
        st.rerun()
    
    # Paso 4: Confirmación
    elif st.session_state.checkout_step == 4:
        st.balloons()
        
        st.success("✅ ¡Pago exitoso!")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image("https://via.placeholder.com/200x200/10b981/ffffff?text=✓", width=150)
        with col2:
            st.markdown(f"""
                <h3>Orden confirmada</h3>
                <p><strong>Número de orden:</strong> #ORD-{random.randint(10000, 99999)}</p>
                <p><strong>ID de pago Haulmer:</strong> {st.session_state.payment_id}</p>
                <p><strong>Total pagado:</strong> ${final_total:,.0f}</p>
                <p><strong>Fecha estimada de entrega:</strong> {(datetime.now() + timedelta(hours=2)).strftime('%H:%M')}</p>
            """, unsafe_allow_html=True)
        
        # Timeline de entrega
        st.subheader("📍 Seguimiento de tu orden")
        
        timeline_cols = st.columns(4)
        steps = [
            ("✅", "Pago confirmado", "Completado"),
            ("🛒", "Preparando pedido", "En curso"),
            ("🚚", "En camino", "Pendiente"),
            ("📦", "Entregado", "Pendiente")
        ]
        
        for i, (icon, label, status) in enumerate(steps):
            with timeline_cols[i]:
                st.markdown(f"""
                    <div style="text-align: center; padding: 1rem; 
                         background: {'#d1fae5' if status == 'Completado' else '#fef3c7' if status == 'En curso' else '#f3f4f6'};
                         border-radius: 10px;">
                        <div style="font-size: 2rem;">{icon}</div>
                        <p style="font-size: 0.8rem; font-weight: 600;">{label}</p>
                        <span style="font-size: 0.7rem; color: #6b7280;">{status}</span>
                    </div>
                """, unsafe_allow_html=True)
        
        # Información de delivery
        if needs_uber:
            st.info("""
                🚗 **Uber Direct asignado**
                - Driver: Carlos M.
                - Vehículo: Toyota Yaris Rojo
                - Patente: ABC-123
                - Llegada estimada: 45 minutos
            """)
        
        if st.button("🏠 Volver al inicio", type="primary"):
            st.session_state.cart = []
            st.session_state.checkout_step = 1
            st.session_state.current_view = "chat"
            st.session_state.messages = []
            st.rerun()

def render_sidebar():
    """Renderiza sidebar con carrito y info"""
    
    with st.sidebar:
        st.title("🛒 Tu Carrito")
        
        if st.session_state.cart:
            total = 0
            for item in st.session_state.cart:
                item_total = item["price"] * item["quantity"]
                total += item_total
                
                st.markdown(f"""
                    <div class="cart-item">
                        <div>
                            <strong>{item['name']}</strong>
                            <br><small>{SUPERMARKETS[item['supermarket']]['name']}</small>
                        </div>
                        <div style="text-align: right;">
                            <strong>${item_total:,.0f}</strong>
                            <br><small>x{item['quantity']}</small>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            st.divider()
            st.markdown(f"<h3 style='text-align: right; color: #059669;'>${total:,.0f}</h3>", unsafe_allow_html=True)
            
            if st.button("💳 Pagar ahora", type="primary", use_container_width=True):
                st.session_state.current_view = "checkout"
                st.rerun()
            
            if st.button("🗑️ Vaciar carrito", use_container_width=True):
                st.session_state.cart = []
                st.rerun()
        else:
            st.info("Tu carrito está vacío")
            st.caption("Agrega productos desde el chat")
        
        st.divider()
        
        # Info de servicios
        st.caption("**Servicios integrados:**")
        st.caption("🤖 OpenAI GPT-4")
        st.caption("💳 Haulmer Payments")
        st.caption("🚚 Uber Direct")
        st.caption("🍳 Spoonacular Recipes")
        
        st.divider()
        st.caption("v1.0.0 - Demo")

def main():
    """Función principal"""
    
    # Header
    st.markdown('<h1 class="main-header">🛒 Smart Shopping Agent</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Powered by AI • Haulmer • Uber Direct</p>', unsafe_allow_html=True)
    
    # Renderizar sidebar
    render_sidebar()
    
    # Vista principal
    if st.session_state.current_view == "chat":
        # Chat container
        render_chat()
        
        # Input area
        st.divider()
        
        col1, col2 = st.columns([6, 1])
        
        with col1:
            user_input = st.text_input(
                "Escribe tu mensaje...",
                key="chat_input",
                placeholder="Ej: 'Tengo papas y crema' o 'Busca leche'",
                label_visibility="collapsed"
            )
        
        with col2:
            if st.button("🎤", help="Entrada de voz", key="voice_btn"):
                st.session_state.is_recording = not st.session_state.is_recording
                if st.session_state.is_recording:
                    st.info("🎤 Grabando... (simulado)")
                    time.sleep(2)
                    # Simular transcripción
                    user_input = "Tengo papas y crema qué puedo cocinar"
                    st.session_state.is_recording = False
        
        # Procesar input
        if user_input:
            # Agregar mensaje del usuario
            add_message("user", user_input)
            
            # Simular procesamiento
            simulate_typing()
            
            # Obtener respuesta del agente
            response = process_user_input(user_input)
            
            # Agregar respuesta del agente
            add_message(
                "agent",
                response["message"],
                response["type"],
                response
            )
            
            # Limpiar input
            st.rerun()
    
    elif st.session_state.current_view == "checkout":
        render_checkout()

if __name__ == "__main__":
    main()