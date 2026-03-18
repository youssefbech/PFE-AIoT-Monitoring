import pandas as pd

file_path = "/mnt/c/Users/MSI/Downloads/bldc_predictive_maintenance_dataset (1) (1).csv"
df = pd.read_csv(file_path)

print(df.head())
print(df.info())



#featurs



target_column = "Fault_Label"   # <-- à modifier si nécessaire

X = df.drop(columns=[target_column])
y = df[target_column]



#normalization entre -1 et 1




from sklearn.preprocessing import MinMaxScaler

scaler_neg1_1 = MinMaxScaler(feature_range=(-1, 1))

X_normalized = scaler_neg1_1.fit_transform(X)

X_normalized = pd.DataFrame(X_normalized, columns=X.columns)

print("Normalisation [-1,1] :")
print(X_normalized.head())
X_normalized[target_column] = y
X_normalized.to_csv("dataset_normalized.csv", index=False)

#standardisation


from sklearn.preprocessing import StandardScaler

scaler_standard = StandardScaler()

X_standardized = scaler_standard.fit_transform(X)

X_standardized = pd.DataFrame(X_standardized, columns=X.columns)

print("Standardisation :")
print(X_standardized.head())

X_standardized[target_column] = y
X_standardized.to_csv("dataset_standardized.csv", index=False)

#test

print("Min après normalisation :", X_normalized.min().min())
print("Max après normalisation :", X_normalized.max().max())

print("Moyenne après standardisation :")
print(X_standardized.mean())

print("Ecart-type après standardisation :")
print(X_standardized.std())