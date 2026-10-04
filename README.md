# 🍳 AI Cooking Assistant & Smart Chef

An intelligent culinary assistant that helps you turn whatever ingredients you already have at home into mouth-watering, chef-quality recipes!

---

## ✨ Features

- 📸 **Multimodal Vision Ingredient Recognition**: Upload a photo of your fridge, pantry, or countertop ingredients to automatically detect items using Google Gemini Vision.
- ✍️ **Interactive Pantry Builder**: Add custom pantry items, choose from common kitchen staples (olive oil, garlic, eggs, etc.), or remove detected items with a single click.
- 🥗 **Dietary & Meal Customization**: Filter recipes by dietary goals (*Vegetarian, Vegan, High Protein, Gluten-Free, Keto, Quick <20 mins*) and meal types (*Dinner, Lunch, Breakfast, Dessert*).
- 🍝 **One-Click Quick Presets**: Test instantly with delicious pre-configured pantries (*Italian Pasta Night, Greek Bowl, Power Breakfast Skillet, Wok Stir-Fry*).
- 💡 **Offline / Demo Mode**: Full functionality and recipe generation even without an external API key.
- ⬇️ **Export Recipes**: Download generated recipes directly as formatted Markdown.

---

## 🚀 Quick Start

### 1. Setup Virtual Environment
```powershell
uv venv --python 3.11
uv pip install -r requirements.txt --python .venv\Scripts\python.exe
```

### 2. Configure Environment (Optional)
Copy `.env.example` to `.env` and add your API keys:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```
*(You can also paste your Gemini API key directly into the sidebar at runtime).*

### 3. Run the Application
```powershell
.\.venv\Scripts\streamlit.exe run streamlit_app.py --server.port 8502
```
Then open **http://localhost:8502** in your browser.
