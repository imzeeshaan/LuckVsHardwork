# Luck vs Hard Work Simulation

An interactive simulation exploring the roles of luck and hard work in determining success over time. This project uses Streamlit to provide real-time visualization and parameter adjustment.

## 🔗 Repository
[github.com/imzeeshaan/LuckVsHardwork](https://github.com/imzeeshaan/LuckVsHardwork)

## 💡 Overview

This simulation compares two groups:
- **Hard Work Only**: Success determined purely by effort
- **Hard Work + Luck**: Success influenced by both effort and random events

The model uses compound growth to demonstrate how these factors affect outcomes over time.

## ⚡ Quick Start

1. Clone the repository:
```bash
git clone https://github.com/imzeeshaan/LuckVsHardwork.git
cd LuckVsHardwork
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Launch the app:
```bash
streamlit run CP.py
```

## 🎮 Interactive Parameters

### Core Settings
- **Population**: Choose between 1,000-20,000 individuals
- **Time Period**: Simulate 5-50 years
- **Hard Work Impact (α)**: Set influence of effort (0.0-0.5)
- **Luck Impact (β)**: Set influence of random events (0.0-0.5)

### Distribution Controls
- **Mean Hard Work (μ)**: Average effort level (0.0-1.0)
- **Hard Work Variation (σ_H)**: Spread in effort (0.01-0.5)
- **Luck Variation (σ_L)**: Spread in random events (0.01-0.5)

## 📊 Pre-set Scenarios

1. **Pure Merit System**
   - Emphasizes hard work (α=0.4)
   - Minimal luck influence (β=0.1)

2. **Luck-Dominated**
   - Strong luck impact (β=0.4)
   - Moderate effort reward (α=0.1)

3. **High Volatility**
   - Maximum randomness
   - Equal effort/luck weights

4. **Long-Term Study**
   - Extended timeframe (50 years)
   - See compounding effects

## 🔧 Technical Details

### Project Structure
```
LuckVsHardwork/
├── CP.py              # Main simulation
├── requirements.txt   # Dependencies
├── README.md         # Documentation
├── Procfile          # Deployment settings
├── .gitignore
└── .streamlit/
    └── config.toml   # Streamlit settings
```

### Mathematical Model
The simulation uses compound growth formulas:
- Hard Work Only: `Success *= (1 + α * HardWork)`
- Hard Work + Luck: `Success *= (1 + α * HardWork + β * Luck)`

Where:
- `HardWork` ~ N(μ, σ_H²)
- `Luck` ~ N(0, σ_L²)

## 🚀 Deployment

### Run on Streamlit Cloud
1. Visit [share.streamlit.io](https://share.streamlit.io)
2. Connect to GitHub
3. Select `imzeeshaan/LuckVsHardwork`
4. Choose `CP.py` as main file

### Local Development
1. Fork the repository
2. Create feature branch:
```bash
git checkout -b feature/your-feature
```
3. Test locally:
```bash
streamlit run CP.py
```
4. Submit pull request

## 📄 License
MIT License - See LICENSE file

## 🤝 Contributing
Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 🙏 Acknowledgments
Built with:
- [Streamlit](https://streamlit.io/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Pandas](https://pandas.pydata.org/)

