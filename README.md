# 🌈📸 OBF-Design

This framework learns optimal **Gaussian Optical Bandpass Filters** that transform high-dimensional spectral signatures into compact, information-rich representations 🌟

> *Deep Gaussian Optical Bandpass Filter Design for Fermentation Index Estimation in Cocoa Beans* 🔬💫

A powerful deep learning framework for **learnable Optical Bandpass Filter Design**! 🎯 Transform spectral signatures into meaningful insights with AI-optimized Gaussian filters 📊✨

## 🎨 Core Innovation

**FilterDesign** is the ⭐ **core** ⭐ of this repository!  
It learns Gaussian optical filters that reduce spectral dimensionality while preserving the most discriminative features 🌟

🔬 **Technical approach:**
- 📡 Processes raw spectral data (hundreds to thousands of bands)
- 🧠 Learns optimal Gaussian filter parameters (μ, σ) 
- ✂️ Reduces dimensionality to physically feasible spectral bands
- 🎯 Achieves state-of-the-art performance with significantly fewer spectral inputs ✨

## 🚀 Quick Start


### 🏃‍♀️ Run FilterDesign with 6 Gaussian filters
```bash
python deep_learning.py --mode filter_design --learned-bands 6 --epochs 10
```

### 🔄 Compare with baseline (all spectral bands, no filtering)
```bash
python deep_learning.py --mode baseline --epochs 10
```

### 🎲 Experiment with different filter counts
```bash
python run.py  # Tests 3, 6, 11 filters automatically
```

## 🛠️ Available Methods

| Mode | Description |
|------|-------------|
| `filter_design` | 🌟 **Primary method** - Learnable Gaussian optical filters |
| `band_selection` | Learnable binary spectral band selection |
| `binary_band_selection` | Hard binary spectral band selection |
| `baseline` | Standard full-spectrum approach (no filtering) |

## 🏗️ Technical Architecture

- 🧪 **FilterDesign**: Gaussian optical filters with learnable μ & σ  
- 🤖 **Multiple Backbones**: SpectraNet, CNN, LSTM, Transformer, SpectralFormer  
- 📈 **Two-Stage Training**: Joint optimization + filter parameter freezing  
- 🎛️ **FWHM Constraints**: Bandwidth limited to physically feasible range (8.25–41.25 nm)

## 📊 Performance Benefits

- 🎯 **Spectral Efficiency**: Drastically reduces dimensionality with minimal performance loss  
- 💡 **Interpretability**: Highlights relevant spectral regions linked to the task  
- 🔧 **Modularity**: Compatible with diverse neural architectures  
- 📈 **Validated**: Demonstrated superiority over state-of-the-art band selection methods  


*Bringing the magic of learnable optics to spectral analysis* 💖  
---
*🌸 Keep learning, keep filtering! 🌸*
