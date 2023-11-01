// We can just plop our data/JSON in this file lol
const DATA = {
  Questions: [
    {
      Questions: ['What does pd.read_csv("Classified Data",index_col=0) do?'],
      Chatgpt_response: [
        'This line is reading a CSV file named "Classified Data" and setting the first column as the index of the DataFrame.',
      ],
      Lines: [6, 6],
    },
    {
      Questions: ["What does df.head() do?"],
      Chatgpt_response: [
        "df.head() displays the first few rows of the DataFrame df. It's useful for getting a quick overview of the data.",
      ],
      Lines: [7, 7],
    },
    {
      Questions: [
        "What does scaled_features = scaler.transform(df.drop('TARGET CLASS',axis=1)) do?",
      ],
      Chatgpt_response: [
        "This line scales the features of the DataFrame, excluding the 'TARGET CLASS' column, using the previously fitted scaler.",
      ],
      Lines: [11, 11],
    },
    {
      Questions: ["Explain this code"],
      Chatgpt_response: [
        "It creates a new DataFrame df_feat with the scaled features and uses the original column names except for the last column, which is the 'TARGET CLASS'.",
      ],
      Lines: [13, 13],
    },
    {
      Questions: ["What is the meaning of test_size=0.30?"],
      Chatgpt_response: [
        "It means that 30% of the data will be used for testing the model, and 70% will be used for training.",
      ],
      Lines: [17, 17],
    },
    {
      Questions: ["Explain this code"],
      Chatgpt_response: [
        "It initializes a KNN classifier with n_neighbors=1, meaning it will consider only the closest neighbor when making predictions.",
      ],
      Lines: [48, 48],
    },
    {
      Questions: ["Explain this code"],
      Chatgpt_response: [
        "This line fits the KNN classifier to the training data, allowing it to learn from the features and corresponding labels.",
      ],
      Lines: [22, 22],
    },
    {
      Questions: ["Explain this code"],
      Chatgpt_response: [
        "It uses the trained KNN model to make predictions on the test data.",
      ],
      Lines: [24, 24],
    },
    {
      Questions: [
        "Explain this code",
        "How does changing the value of n_neighbors affect the behavior of the K-Nearest Neighbors model?",
      ],
      Chatgpt_response: [
        "This line initializes a K-Nearest Neighbors classifier with n_neighbors=1, meaning it considers only the closest neighbor when making predictions.",
        "Changing n_neighbors alters the number of neighboring data points considered for classification. A lower value (like 1) makes the model more sensitive to outliers, while a higher value smoothens the decision boundary.",
      ],
      Lines: [48, 48],
    },
    {
      Questions: [
        "Explain this code",
        "What would happen if you didn't fit the K-Nearest Neighbors model to the training data before making predictions?",
      ],
      Chatgpt_response: [
        "This line fits the K-Nearest Neighbors classifier to the training data (X_train and y_train), allowing it to learn from the features and corresponding labels.",
        "If you didn't fit the model to the training data, it would not have learned from the data, and making predictions would not be meaningful.",
      ],
      Lines: [22, 22],
    },
    {
      Questions: [
        "What is the significance of the parameters algorithm, leaf_size, metric, metric_params, n_jobs, p, and weights?",
        "Can you explain what each of the parameters does in the context of the K-Nearest Neighbors classifier?",
      ],
      Chatgpt_response: [
        "These are hyperparameters and configurations for the K-Nearest Neighbors classifier. They control aspects like the algorithm used, distance metric, number of parallel jobs, etc.",
        "Certainly, algorithm selects the algorithm to use for nearest neighbors search, leaf_size is the number of points at which the algorithm switches to brute-force, metric defines the distance metric, metric_params are additional keyword arguments for the metric function, n_jobs is the number of parallel jobs to run for neighbors search, p is the power parameter for the Minkowski metric, and weights specifies the weight function used in prediction.",
      ],
      Lines: [23, 23],
    },
    {
      Questions: [
        "Explain this code",
        "What kind of data is stored in the variable pred after making predictions?",
      ],
      Chatgpt_response: [
        "This line uses the trained K-Nearest Neighbors model (knn) to make predictions on the test data (X_test) and stores the results in the variable pred.",
        "The variable pred will contain the predicted labels for the test data based on the K-Nearest Neighbors model.",
      ],
      Lines: [24, 24],
    },
    {
      Questions: [
        "Explain this code",
        "What other preprocessing techniques might be used in combination with standardization?",
        "Can you provide an example of when one-hot encoding might be more suitable than standardization for preprocessing?",
      ],
      Chatgpt_response: [
        "This line initializes an instance of the StandardScaler class, which will be used to scale the features.",
        "Other techniques might include one-hot encoding for categorical variables, handling missing values, and feature selection.",
        "One-hot encoding is typically used for categorical variables to represent them as binary values. It's not used for continuous numerical features.",
      ],
      Lines: [10, 10],
    },
    {
      Questions: [
        "Explain this code",
        "Is it possible to use a different scaler besides StandardScaler for feature scaling?",
        "What are the differences between StandardScaler and MinMaxScaler in terms of how they scale the data?",
      ],
      Chatgpt_response: [
        "This line fits the StandardScaler to the data by computing the mean and standard deviation needed for later scaling.",
        "Yes, scikit-learn provides various scalers like MinMaxScaler, RobustScaler, etc. depending on the specific needs of the dataset.",
        "StandardScaler standardizes the data to have a mean of 0 and a standard deviation of 1, while MinMaxScaler scales the data to a specified range, typically [0, 1].",
      ],
      Lines: [11, 11],
    },
    {
      Questions: [
        "Explain this code",
        "What could be the potential consequences if you skipped the step of fitting the StandardScaler before transforming the data?",
        "Are there scenarios where you might want to fit the scaler without transforming the data immediately?",
      ],
      Chatgpt_response: [
        "This line transforms the features by applying the previously computed mean and standard deviation to standardize the data.",
        "If you skip fitting the StandardScaler, the scaler won't have the necessary information (mean and standard deviation) to properly scale the features, leading to incorrect scaling results.",
        "Yes, fitting the scaler separately can be useful when you want to standardize training and test sets using the same scaler to ensure consistency.",
      ],
      Lines: [12, 12],
    },
    {
      Questions: [
        "Explain this code",
        "Can you explain why it's beneficial to create a new DataFrame (df_feat) with the scaled features instead of directly modifying the original DataFrame (df)?",
        "What are some potential drawbacks of modifying the original DataFrame in place instead of creating a new one like df_feat?",
      ],
      Chatgpt_response: [
        "It creates a new DataFrame (df_feat) with the scaled features and uses the original column names except for the last column, which is the 'TARGET CLASS'.",
        "Creating a new DataFrame keeps the original data intact, allowing you to compare and analyze the scaled and unscaled data. It also ensures that the original data remains available for reference or further processing.",
        "Modifying the original DataFrame could result in data loss, making it challenging to compare the original and scaled data or recover the original data.",
      ],
      Lines: [13, 13],
    },
    {
      Questions: [
        "Explain this code",
        "How might you use the information from df_feat.head() to assess the effect of standardization on the features?",
        "Are there any visualizations or statistical analyses you could perform on the output of df_feat.head() to better understand the impact of standardization?",
      ],
      Chatgpt_response: [
        "df_feat.head() displays the first few rows of the DataFrame df_feat. It's useful for getting a quick overview of the scaled features.",
        "df_feat.head() provides an initial look at the scaled features, which can help in assessing their distribution and spotting any noticeable changes resulting from standardization.",
        "Yes, you can create histograms or box plots to observe the distribution of the scaled features and calculate summary statistics to see how they compare to the original data.",
      ],
      Lines: [14, 14],
    },
    {
      Questions: ["Explain this code"],
      Chatgpt_response: [
        "This line is used to create a line plot. It plots the error_rate values against the range of k values (from 1 to 39) to visualize how the error rate changes with different values of k.",
      ],
      Lines: [40, 40],
    },
    {
      Questions: ["What does range(1,40) represent in this context?"],
      Chatgpt_response: [
        "range(1,40) generates a sequence of numbers from 1 to 39, which corresponds to the range of k values being considered for the K-Nearest Neighbors model.",
      ],
      Lines: [40, 40],
    },
    {
      Questions: [
        "How does changing the parameters color, linestyle, marker, markerfacecolor, and markersize affect the appearance of the plot?",
      ],
      Chatgpt_response: [
        "These parameters control the color, line style, marker style, marker face color, and marker size used in the plot, allowing for customization of the visual representation.",
      ],
      Lines: [40, 40],
    },
    {
      Questions: [
        "What libraries are being imported in the first few lines of the code?",
      ],
      Chatgpt_response: [
        "These lines are importing necessary libraries for data manipulation and visualization, including pandas, seaborn, matplotlib, and numpy.",
      ],
      Lines: [1, 4],
    },
    {
      Questions: ["What is the purpose of StandardScaler in this code?"],
      Chatgpt_response: [
        "StandardScaler is used to standardize the features by removing the mean and scaling to unit variance.",
      ],
      Lines: [9, 12],
    },
    {
      Questions: ["Explain this code"],
      Chatgpt_response: [
        "It's iterating through different values of n_neighbors to find the one that minimizes the error rate.",
      ],
      Lines: [32, 36],
    },
    {
      Questions: [
        "How does setting K=23 in the KNN classifier differ from the previous case with K=1?",
      ],
      Chatgpt_response: [
        "Setting K=23 means that the KNN classifier considers 23 closest neighbors when making predictions, which can result in a smoother decision boundary and may be less sensitive to individual data points.",
      ],
      Lines: [50, 72],
    },
    {
      Questions: [
        "What is the purpose of a train-test split in machine learning?",
      ],
      Chatgpt_response: [
        "A train-test split is used to evaluate the performance of a machine learning model. It separates the data into two sets: one for training the model and the other for testing its performance.",
      ],
      Lines: [1, 72],
    },
    {
      Questions: ["What is the purpose of a K-Nearest Neighbors Classifier?"],
      Chatgpt_response: [
        "The K-Nearest Neighbors (KNN) classifier is a type of supervised learning algorithm used for both classification and regression tasks. It classifies an object based on its neighboring objects in the feature space.",
      ],
      Lines: [1, 72],
    }
  ],
};

// Test code from: https://github.com/vivek2319/K-Nearest-Neighbors
const TEST_CODE = `import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("Classified Data",index_col=0)
df.head()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df.drop('TARGET CLASS',axis=1))
scaled_features = scaler.transform(df.drop('TARGET CLASS',axis=1))
df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])
df_feat.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(scaled_features,df['TARGET CLASS'], test_size=0.30)
                
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=1, n_neighbors=1, p=2, weights='uniform')
pred = knn.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(y_test,pred))

error_rate = []

for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10,6))

plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',markerfacecolor='red', markersize=10)

plt.title('Error Rate vs. K Value')

plt.xlabel('K')

plt.ylabel('Error Rate')

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print('WITH K=1')
print('\\n')
print(confusion_matrix(y_test,pred))
print('\\n')
print(classification_report(y_test,pred))

# WITH K=1

knn = KNeighborsClassifier(n_neighbors=23)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print('WITH K=23')
print('\\n')
print(confusion_matrix(y_test,pred))
print('\\n')
print(classification_report(y_test,pred))

# WITH K=23
`;
