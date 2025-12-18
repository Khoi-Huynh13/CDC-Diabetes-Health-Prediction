# Diabetes Risk Prediction Workflow
<p align="justify">
This project builds and evaluates machine learning models to predict whether or not a patient has diabetes risk using lab test results and key health indicators. The workflow covers data preprocessing, exploratory data analysis, and feature engineering to improve model performance and generalizability. Moreover, to address the problem of class imbalance in the dataset, SMOTE (Synthetic Minority Over-sampling Technique) is applied to enhance minority-class representation. Multiple supervised models are trained and evaluated using a variety of metrics for balanced performance assessment. See the below sections for more details. Cross-validation is used during training and hyperparameter tuning to reduce overfitting and ensure robust generalization.
</p>

<p align="center">
  <img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/3f05999f-ac02-4857-8475-b21097115fba" />
</p>

# Data Dictionary
| Feature | Description | Example Value |
| :--------: | :------: | :--------: |
| ID | Patient ID | 5 |
| Diabetes_binary | 0 = no diabetes 1 = prediabetes or diabetes | 1 |
| HighBP | Whether the patient has high blood pressure. 0 = no high BP 1 = high BP | 0 |
| HighChol | Whether the patient has high cholesterol. 	0 = no high cholesterol 1 = high cholesterol	| 1 |
| CholCheck | Whether the patient has their cholesterol level checked within the last 5 years. 0 = no cholesterol check in 5 years 1 = yes cholesterol check in 5 years | 0 |
| BMI | Patient's body mass index | 38.0 |
| Smoker | Whether the patient has smoked at least 100 cigarettes in their entire life. [Note: 5 packs = 100 cigarettes] 0 = no 1 = yes | 1 |
| Stroke | Whether the patient has ever been told they had a stroke. 0 = no 1 = yes | 1 |
| HeartDiseaseorAttack | Whether the patient has coronary heart disease (CHD) or myocardial infarction (MI). 0 = no 1 = yes | 1 |
| PhysActivity | Whether the patient has any physical activity in past 30 days - not including job. 0 = no 1 = yes | 0 |
| Fruits | Whether the patient consumes fruit 1 or more times per day. 0 = no 1 = yes | 1 |
| Vegetable | Whether the patient consumes vegetables 1 or more times per day 0 = no 1 = yes | 0 |
| HvyAlcoholConsump | Whether the patient is a heavy drinker. Heavy drinkers (adult men having more than 14 drinks per week and adult women having more than 7 drinks per week) 0 = no 1 = yes. | 0 | 
| AnyHealthcare | Whether the patient has any kind of health care coverage, including health insurance and prepaid plans. 0 = no 1 = yes. | 1 |
| NoDocbcCost | Whether there was a time in the past 12 months when the patient needed to see a doctor but could not because of cost? 0 = no 1 = yes. | 0 |
| GenHlth | Patient's general health on a scale of 1-5. 1 = excellent 2 = very good 3 = good 4 = fair 5 = poor | 4 |
| MentHlth | How many days during the past 30 days was the patient's mental health not good, including but not limited to stress, depression, and problems with emotions? | 17 |
| PhysHlth | How many days during the past 30 days was the patient's physical health not good, including but not limited to physical illness and injury? | 29 |
| DiffWalk | Whether the patient has any difficulty walking or climbing stairs. 0 = no 1 = yes | 0 |
| Sex | Patient's sex. | Female |
| Age | Patient's age (equi width binning in 13 bins, width = 5) | 12 | 
| Education | Patient's education (6 categories). 1 = Never attended school or only kindergarten 2 = Grades 1 through 8 (Elementary) 3 = Grades 9 through 11 (Some high school) 4 = Grade 12 or GED (High school graduate) 5 = College 1 year to 3 years (Some college or technical school) 6 = College 4 years or more (College graduate). | 3 |
| Income | Patient's income on a scale 1-8. 1 = less than $10,000 5 = less than $35,000 8 = $75,000 or more | 5 |

Source: https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators

# Exploratory Data Analysis (EDA)
Below are the insights gathered that serves as a preliminary test to determine which features significantly influence the risk of getting diabetes.

+ People in older age groups are more likely to have diabetes.
<img width="500" height="340" alt="image" src="https://github.com/user-attachments/assets/7dee2a8f-0145-4c39-872c-55a61853a388" />

+ People in high income groups are less likely to have diabetes.
<img width="500" height="340" alt="image" src="https://github.com/user-attachments/assets/bb1409db-bb87-4832-9de5-1db88a401247" />

+ Patient's BMI potentially has a linear relationship with diabetes logit as shown in the the empirical logit plot below. Therefore, this feature can be appropriate in a logistic regression model.
<img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/67b6b761-fb39-4d7d-be8f-71d831f9ecf0" />

+ High blood pressure and cholesterol levels are key indicators for diabetes.
<img width="500" height="350" alt="image" src="https://github.com/user-attachments/assets/864f4e9b-4e66-4cec-8547-0b9a4d2bccf9" />
<img width="500" height="350" alt="image" src="https://github.com/user-attachments/assets/ca1a2cfa-bcb8-43db-889d-bda75e6cd2aa" />

+ Whether the patient has CHD/MI or not is also a potential diabetes indicator, but the significant difference between the sample sizes of the 2 categories does undermine the reliability of this claim.
<img width="500" height="350" alt="image" src="https://github.com/user-attachments/assets/e0d6dd41-d9c7-410e-a47f-316cdecf7d63" />

# Feature Selection
<p align="justify">
After the preliminary feature screening via EDA, in order to concretely select features that are most relevant and important to the target, we performed two related statistical tests which are the <i>Chi-Square Test of Independence</i> and <i>Cramer's V</i>. The former test is to determine whether or not the target variable (diabetes) is related to a categorical feature. The latter test is used measure how strongly two categorical features are associated or in other words, it quantifies the strength of the relationship between two categorical features. This is determined by using a calculated score often referred to as "effect size". For more detail on how this score is calculated, refer to: https://www.ibm.com/docs/en/cognos-analytics/12.0.x?topic=terms-cramrs-v. In general, the interpretation of the effect size and the different thresholds are as follows:
</p>

| Effect size (ES) | Interpretation |
| :--------: | :--------: |
| ES ≤ 0.2 | The result is weak. Although the result is statistically significant, the fields are only weakly associated. |
| 0.2 < ES ≤ 0.6 | The result is moderate. The fields are moderately associated. |
| ES > 0.6 | The result is strong. The fields are strongly associated. |

Performing the two mentioned statistical tests to all the available categorical features gives the result as seen in the bar chart below. We only select features with ES > 0.2. Therefore, the list of remaining features consists of:
+ HighBP
+ HighChol
+ BMI
+ HeartDieasesorAttack
+ GenHlth
+ DiffWalk
+ Age
+ Income
+ PhysHlth_Bin
<img width="1364" height="269" alt="image" src="https://github.com/user-attachments/assets/4bbc9a39-4d30-4ec2-a9c6-4ec161e1b6a4" />





