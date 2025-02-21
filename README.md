# Barbell Exercise Repetition Counter

This machine learning project is designed to predict and analyze fitness levels using several health-related metrics. We use multiple machine learning models to determine the effectiveness and accuracy of multiple feature sets in predicting fitness scores. The project enables people to monitor their fitness progress and make informed health decisions.


**PROJECT OVERVIEW**

1) Converting raw data, reading CSV files, spliting and cleaning data
2) Visualizing data, Plotting time series data.
3) Outlier detection
4) Feature engineering, LowPass filter, Principal Component Analysis, Clustering
5) Chosing appropriate model
6) Counting repetitions


## Features & Capabilities
- Feature Selection & Engineering**: Analyzed multiple feature sets to improve model performance.
- Model Training & Evaluation**: Implemented various machine learning models to compare predictive accuracy.
- Visualization & Analysis**: Generated comparative accuracy charts for different feature sets and models.
- Optimized Model Deployment**: Selected the best-performing model for potential real-time fitness predictions.

## Tools & Technologies Used
- **Programming Language**: Python
- **Libraries**: Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn
- **Models Used**:
  - Random Forest (RF)
  - Decision Tree (DT)
  - Neural Networks (NN)
  - Naive Bayes (NB)
  - K-Nearest Neighbors (KNN)
- **Visualization**: Matplotlib, Seaborn

## Model Performance Analysis
The project evaluates five different machine learning models using various feature sets to determine the best fitness predictor. Below is a summary of accuracy results:

![image](https://github.com/user-attachments/assets/91b5a156-196e-432e-a12f-e16df4099791)


### **Observations:**
- Random Forest (RF) and Neural Networks (NN) achieved the highest accuracy.
- KNN performed poorly with some feature sets, making it unsuitable for deployment.
- Feature Set 4 and Selected Features provided the most consistent and high-performing results.
- Naive Bayes (NB) showed moderate accuracy but underperformed compared to RF and NN.

Based on these findings, **Random Forest and Neural Networks were the top contenders for fitness prediction**, with RF being the best choice for deployment.

## Project Structure
```
|-- MetaMotion/              # Contains dataset files
|-- reports/figures/        # Stores figures and visualizations
|-- src/
    |-- data/
        |-- features/       # Feature engineering scripts (e.g., DataTransformation.py)
        |-- interim/        # Intermediate data storage
        |-- models/         # Model training and learning scripts (e.g., LearningAlgorithms.py, train_model.py)
        |-- visualization/  # Scripts for data visualization
        |-- make_dataset.py # Script for preprocessing or dataset creation
|-- README.md               # Project documentation
|-- environment.yml         # Dependencies and environment setup
```



## Results & Conclusion
- The **best model (Random Forest)** achieved an accuracy of **97.5204%**, making it highly effective for fitness tracking.
- Feature selection played a significant role in improving model accuracy.
- Future improvements could include real-time fitness tracking integration with IoT devices.



## Future Enhancements
- Integration with wearable fitness devices** for real-time predictions.
- Deep learning implementation** to improve accuracy further.
- Web/Mobile App Deployment** to make fitness tracking more user-friendly.

---
✅ **Author:** Gurudatta Bidkar  

📧 **Contact:** gurudattaxd@gmail.com

🔗 **GitHub Repository:** https://github.com/gurudattaxd/Machine-learning-project-Fitness-Tracker-













