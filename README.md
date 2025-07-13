# PyTorch Practice Repository 🔥

A comprehensive collection of PyTorch tutorials and implementations covering fundamental concepts to advanced deep learning techniques. This repository is designed to provide hands-on experience with PyTorch through practical examples and real-world applications.

## 📚 Repository Structure

### 1. Tensor Operations (`1.Tensor/`)
- **`basic_tensor_cpu.ipynb`** - Introduction to tensor creation and basic operations on CPU
- **`basic_tensor_gpu.ipynb`** - GPU tensor operations and CUDA basics
- **`tensors_in_pytorch.ipynb`** - Comprehensive tensor manipulation techniques

### 2. Automatic Differentiation (`2.Autograd/`)
- **`derivative.ipynb`** - Understanding derivatives and gradients
- **`derivative_neural_network.ipynb`** - Derivatives in neural network context
- **`pytorch_autograd.ipynb`** - PyTorch's automatic differentiation system

### 3. Building Neural Networks (`3.BuildingNeuralNetwork/`)
- **`brest_cancer.ipynb`** - Binary classification using breast cancer dataset
- **`pytorch_training_pipeline.ipynb`** - Complete training pipeline from scratch

### 4. Neural Network Modules (`4.NNModule/`)
- **`simple_nn.ipynb`** - Basic neural network implementation
- **`hiddenlayer_nn.ipynb`** - Multi-layer neural networks
- **`pytorch_nn_module.ipynb`** - Using PyTorch's nn.Module
- **`pytorch_training_pipeline_using_nn_module.ipynb`** - Training with nn.Module
- **`brest_cancer.ipynb`** - Breast cancer classification with nn.Module

### 5. Data Loading (`5.DataLoader/`)
- **`dataset_and_dataloader_demo.ipynb`** - Custom datasets and data loaders
- **`pytorch_training_pipeline_using_dataset_and_dataloader.ipynb`** - Complete pipeline with DataLoader
- **`brest_cancer.ipynb`** - Breast cancer classification with custom data loading

### 6. Artificial Neural Networks (`6.ANN/`)
- **`ann_fashion_mnist_pytorch.ipynb`** - Fashion-MNIST classification with ANN
- **`ann_fashion_mnist_pytorch_gpu.ipynb`** - GPU-accelerated Fashion-MNIST
- **`ann_fashion_mnist_pytorch_gpu_optimized.ipynb`** - Optimized GPU implementation
- **`fashion_reduce_overfitting.ipynb`** - Techniques to reduce overfitting
- **`fashion.ipynb`** - Additional Fashion-MNIST experiments
- **Dataset files**: `fashion-mnist_test.csv`, `fmnist_small.csv`

### 7. Hyperparameter Optimization (`7.Optuna/`)
- **`optuna_basics_yt.ipynb`** - Introduction to Optuna framework
- **`ann_fashion_mnist_pytorch_gpu_optimized_optuna.ipynb`** - Hyperparameter tuning with Optuna
- **`test.ipynb`** - Optuna optimization experiments
- **`1907.10902v1.pdf`** - Research paper on hyperparameter optimization

### 8. Convolutional Neural Networks (`8.CNN/`)
- **`cnn_fashion_mnist_pytorch_gpu.ipynb`** - CNN implementation for Fashion-MNIST
- **`test.ipynb`** - CNN experiments and testing
- **Dataset**: `fmnist_small.csv`

### 9. Transfer Learning (`9.Transfer Learning/`)
- **`transfer_learning_fashion_mnist_pytorch_gpu.ipynb`** - Transfer learning with pre-trained models
- **`cnn_optuna.ipynb`** - CNN hyperparameter optimization
- **`test.ipynb`** - Transfer learning experiments
- **Dataset**: `fmnist_small.csv`

### 10. Recurrent Neural Networks (`10.RNN/`)
- **`pytorch_rnn_based_qa_system.ipynb`** - Question-Answering system using RNN
- **`try.ipynb`** - RNN experiments
- **Dataset**: `100_Unique_QA_Dataset.csv`

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch torchvision torchaudio
pip install pandas numpy matplotlib scikit-learn
pip install optuna jupyter
```

### Quick Start

1. Clone this repository:
```bash
git clone https://github.com/yourusername/PyTorch-Practice.git
cd PyTorch-Practice
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start with tensor basics:
```bash
jupyter notebook 1.Tensor/basic_tensor_cpu.ipynb
```

## 📖 Learning Path

### Beginner (Start Here)
1. **Tensor Operations** - Learn PyTorch fundamentals
2. **Autograd** - Understand automatic differentiation
3. **Building Neural Networks** - Create your first neural network

### Intermediate
4. **NN Module** - Master PyTorch's neural network modules
5. **DataLoader** - Efficient data handling and preprocessing
6. **ANN** - Build and optimize artificial neural networks

### Advanced
7. **Optuna** - Hyperparameter optimization techniques
8. **CNN** - Convolutional neural networks for image processing
9. **Transfer Learning** - Leverage pre-trained models
10. **RNN** - Sequential data processing and NLP applications

## 🎯 Key Features

- **Comprehensive Coverage**: From basic tensors to advanced architectures
- **GPU Support**: CUDA implementations for faster training
- **Real Datasets**: Fashion-MNIST, Breast Cancer, Q&A datasets
- **Best Practices**: Industry-standard coding patterns and optimization techniques
- **Hyperparameter Tuning**: Advanced optimization with Optuna
- **Transfer Learning**: Pre-trained model utilization
- **Multiple Architectures**: ANN, CNN, RNN implementations

## 📊 Datasets Used

- **Fashion-MNIST**: Fashion item classification (10 classes)
- **Breast Cancer**: Binary classification for medical diagnosis
- **Custom Q&A Dataset**: 100 unique question-answer pairs for RNN training

## 🛠️ Technologies

- **PyTorch**: Deep learning framework
- **CUDA**: GPU acceleration
- **Optuna**: Hyperparameter optimization
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Matplotlib**: Data visualization
- **Scikit-learn**: Machine learning utilities

## 📈 Performance Optimizations

- GPU acceleration with CUDA
- Batch processing with DataLoader
- Hyperparameter tuning with Optuna
- Transfer learning for faster convergence
- Regularization techniques (Dropout, etc.)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 Notes

- The `notes.pdf` file contains additional theoretical background
- Each notebook is self-contained with detailed explanations
- GPU notebooks require CUDA-compatible hardware
- Start with CPU versions if GPU is not available

## 🔗 Useful Resources

- [PyTorch Official Documentation](https://pytorch.org/docs/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Optuna Documentation](https://optuna.readthedocs.io/)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⭐ Star This Repository

If you find this repository helpful, please give it a star! It helps others discover this resource.

---

**Happy Learning with PyTorch! 🎉**
