# 📈 Stock Price Prediction Web App  

A full-stack machine learning project that predicts future stock prices using an **LSTM neural network** and displays results in a simple **Flask web interface**.  

The project integrates:  
- 📥 Data ingestion from **Alpha Vantage API**  
- 🧹 Data preprocessing and normalization  
- 🧠 LSTM model training in **PyTorch**  
- 📊 Visualization of price history, training/validation splits, loss curves, and predictions  
- 🌐 Frontend search bar for stock ticker symbols with results and plots served via **Flask**  

---

## 🗂 Project Structure  

```bash
STOCK_PRICE_PREDICTOR/
├─ backend/
│  ├─ app.py                # Flask app (routes for training, summary, autocomplete)
│  ├─ models/
│  │  ├─ __init__.py
│  │  ├─ config.py          # Central configuration dictionary
│  │  ├─ utils.py           # Data preprocessing + normalization + plotting utilities
│  │  ├─ lstm_model.py      # LSTM model class + training + prediction wrappers
│  ├─ static/
│  │  ├─ style.css          # Frontend styling
│  │  └─ plots/             # Auto-generated plots (saved here by utils.py)
│  └─ templates/
│     └─ index.html         # Jinja2 HTML template (search form + summary + charts)
├─ requirements.txt          # Python dependencies
├─ README.md                 # This file
└─ project_log.md            # Development notes
```


🚀 Getting Started
1. Clone the repo
git clone https://github.com/your-username/stock-price-predictor.git
cd stock-price-predictor

2. Create a virtual environment & install dependencies
python3 -m venv venv
source venv/bin/activate   # (Windows: venv\Scripts\activate)
pip install -r requirements.txt

3. Get an Alpha Vantage API key
	•	Sign up at Alpha Vantage
	•	Replace "demo" in backend/models/config.py with your key:

"alpha_vantage": {
    "key": "YOUR_API_KEY_HERE",
    "symbol": "IBM",
    "outputsize": "full",
    "key_adjusted_close": "5. adjusted close",
}

4. Run the Flask APP
cd backend
python app.py

🧑‍💻 Development

Configuration
All model, data, and training hyperparameters are stored in backend/models/config.py. Example:

"model": {
    "input_size": 1,
    "num_lstm_layers": 2,
    "lstm_size": 32,
    "dropout": 0.2,
},
"training": {
    "device": "cpu",  # or "cuda"
    "batch_size": 64,
    "num_epoch": 100,
    "learning_rate": 0.01,
}

## 🗂 Code Organization

- `backend/models/utils.py`
  - Data download (Alpha Vantage)
  - Preprocessing (windowing, splits)
  - Normalization (`Normalizer`)
  - Datasets & DataLoaders (`TimeSeriesDataset`)
  - Plot helpers (save `.png` files to `static/plots/`)

- `backend/models/lstm_model.py`
  - `LSTMModel` (PyTorch)
  - `run_epoch` (train/eval step)
  - `train_model` / `train_model_for_symbol` (training orchestration)
  - `predict_next` (single-step inference)

- `backend/app.py` (Flask routes)
  - `GET /` — index page with search bar
  - `GET /summary/<ticker>` — trains/loads model, generates plots, renders template
  - `GET /auto_complete?query=...` — optional ticker search suggestions

- `backend/templates/index.html`
  - Jinja2 template that displays summary + plots (`<img src="/static/plots/...">`)

- `backend/static/`
  - `style.css` — frontend styling
  - `plots/` — **generated images only** (saved by plotting helpers)

 📦 Requirements

See requirements.txt. Major dependencies:
	•	Python 3.9+
	•	Flask
	•	PyTorch
	•	NumPy
	•	Matplotlib
	•	Alpha Vantage API client


👤 Author

Krishang Sharma
	•	📧 Contact: krishangs468@gmail.com
	•	💼 LinkedIn: https://www.linkedin.com/in/krishang-sharma1/
 









