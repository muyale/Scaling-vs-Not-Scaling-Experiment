
# Stroke Prediction: Impact of Data Scaling on Model Performance

## Table of Contents
1. [Introduction](#introduction)
2. [Objective](#objective)
3. [Dataset](#dataset)
4. [Preprocessing](#preprocessing)
    - [Data Scaling](#data-scaling)
    - [Train-Test Split](#train-test-split)
5. [Model](#model)
    - [Random Forest Classifier](#random-forest-classifier)
6. [Methodology](#methodology)
    - [Experiment Setup](#experiment-setup)
    - [Evaluation](#evaluation)
7. [Results](#results)
    - [Accuracy Scores](#accuracy-scores)
    - [Findings](#findings)
8. [Conclusion](#conclusion)
9. [Future Work](#future-work)
10. [How to Run](#how-to-run)
11. [Dependencies](#dependencies)
12. [Contributing](#contributing)
13. [License](#license)
14. [Contact](#contact)

## Introduction
This project aims to analyze the impact of data scaling on the performance of a Random Forest Classifier in predicting stroke diagnoses. Data scaling is a crucial preprocessing step in machine learning that can influence model performance. This study compares the effects of using StandardScaler and MinMaxScaler on the accuracy of the model.

## Objective
To determine whether scaling the data impacts the performance of a Random Forest Classifier in predicting stroke diagnoses.

## Dataset
The dataset contains information on patients diagnosed with stroke. The features include various health and demographic attributes, and the target variable indicates whether the patient had a stroke.

## Preprocessing

### Data Scaling
- **StandardScaler**: Scales the data to have a mean of 0 and a standard deviation of 1.
- **MinMaxScaler**: Scales the data to a fixed range, typically between 0 and 1.

### Train-Test Split
The data is split into training and testing sets using a 80-20 ratio.

## Model

### Random Forest Classifier
A Random Forest Classifier is an ensemble learning method that constructs multiple decision trees and outputs the mode of their predictions. It is robust to overfitting and performs well on various types of data.

## Methodology

### Experiment Setup
1. Load the dataset and preprocess it by scaling the features using StandardScaler and MinMaxScaler.
2. Split the data into training and testing sets.
3. Train a Random Forest Classifier on the training data.
4. Evaluate the model's performance on the testing data over 5 epochs.

### Evaluation
The model's performance is evaluated using accuracy, which is the proportion of correctly classified instances among the total instances.

## Results

### Accuracy Scores
| Epoch | Normal Accuracy (%) | StandardScaler Accuracy (%) | MinMaxScaler Accuracy (%) |
|-------|----------------------|-----------------------------|---------------------------|
| 1     | 94.61                | 94.72                       | 94.52                     |
| 2     | 94.72                | 94.72                       | 94.42                     |
| 3     | 94.72                | 94.72                       | 94.13                     |
| 4     | 94.72                | 94.72                       | 94.72                     |
| 5     | 94.62                | 94.72                       | 94.72                     |

### Findings
- Scaling the data has a minimal impact on the performance of the Random Forest Classifier for stroke prediction.
- StandardScaler shows slightly more stability with consistent accuracy across all epochs.
- Overall, the model demonstrates high accuracy irrespective of scaling, indicating robustness to data scaling for this specific problem.

## Conclusion
The choice of scaling method may not significantly affect model performance for this dataset. However, StandardScaler provides slightly more stable results. Further exploration of different scaling methods and machine learning models could provide additional insights.

## Future Work
- Experiment with other deep learning algorithms like neural networks, CNNs, and RNNs.
- Explore ensemble methods such as Gradient Boosting Machines and stacking.
- Implement hyperparameter tuning to optimize model performance.
- Conduct feature engineering to create new informative features.
- Use data augmentation techniques to enhance the training dataset.

## How to Run
1. Clone the repository: `git clone https://github.com/muyale/stroke-Scaling-vs-Not-Scaling-Experiment.git`
2. Navigate to the project directory: `cd Scaling-vs-Not-Scaling-Experiment`
3. Install the required dependencies: `pip install -r requirements.txt`
4. Run the Jupyter Notebook: `jupyter notebook`

## Dependencies
- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Jupyter Notebook

## Contributing
Contributions are welcome! Please fork the repository and create a pull request with your changes. Ensure your code follows the project's coding standards and includes appropriate tests.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For any questions or inquiries, please contact Edgar at edgarmuyale@gmail.com.

---

