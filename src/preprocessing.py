import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess(file_path):

    df = pd.read_csv(file_path)

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # BMI Category Feature
    def bmi_category(bmi):
        if bmi < 18.5:
            return 0
        elif bmi < 25:
            return 1
        elif bmi < 30:
            return 2
        else:
            return 3

    df['bmi_category'] = df['bmi'].apply(bmi_category)

    le = LabelEncoder()

    df['sex'] = le.fit_transform(df['sex'])
    df['smoker'] = le.fit_transform(df['smoker'])
    df['region'] = le.fit_transform(df['region'])

    return df