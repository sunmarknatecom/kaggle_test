# from todaycode (youtube)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import missingno as msno

question = pd.read_csv('ds_survey_2017/schema.csv')

mcq = pd.read_csv('ds_survey_2017/multipleChoiceResponses.csv', encoding='ISO-8859-1', low_memory=False)

msno.matrix(mcq, figsize=(12,5))
plt.show()

sns.countplot(y='GenderSelect', data=mcq)
plt.show()

con_df = pd.DataFrame(mcq['Country'].value_counts())
con_df['국가'] = con_df.index
con_df.columns = ['응답 수', '국가']
con_df = con_df.reset_index().drop('index', axis=1)
con_df.head(20)

mcq['Age'].describe()

sns.distplot(mcq[mcq['Age'] > 0]['Age'])
plt.show()

sns.countplot(y='FormalEducation', data=mcq)
plt.show()

mcq_major_count = pd.DataFrame(mcq['MajorSelect'].value_counts())
mcq_major_percent = pd.DataFrame(mcq['MajorSelect'].value_counts(normalize=True))
mcq_major_df = mcq_major_count.merge(mcq_major_percent, left_index=True, right_index=True)
mcq_major_df.columns = ['응답 수', '비율']
mcq_major_df

plt.figure(figsize=(6,8))
sns.countplot(y='MajorSelect', data=mcq)
plt.show()

mcq_es_count = pd.DataFrame(mcq['EmploymentStatus'].value_counts())
mcq_es_percent = pd.DataFrame(mcq['EmploymentStatus'].value_counts(normalize=True))
mcq_es_df = mcq_es_count.merge(mcq_es_percent, left_index=True, right_index=True)
mcq_es_df.columns = ['응답 수', '비율']
mcq_es_df

sns.countplot(y='EmploymentStatus', data=mcq)
plt.show()

sns.countplot(y='Tenure', data=mcq)
plt.show()

korea = mcq.loc[(mcq['Country']=='South Korea')]

print('The number of interviewees in Korea: ' + str(korea.shape[0]))

sns.distplot(korea['Age'].dropna())
plt.title('Korean')
plt.show()

pd.DataFrame(korea['GenderSelect'].value_counts())

sns.countplot(x='GenderSelect', data=korea)
plt.title('Korean')
plt.show()

figure, (ax1, ax2) = plt.subplots(ncols=2)
figure.set_size_inches(12,5)
sns.distplot(korea['Age'].loc[korea['GenderSelect']=='Female'].dropna(), norm_hist=False, color=sns.color_palette("Paired")[4], ax=ax1)
plt.title('korean Female')

sns.distplot(korea['Age'].loc[korea['GenderSelect']=='Male'].dropna(), norm_hist=False, color=sns.color_palette("Paired")[0], ax=ax2)
plt.title('korean Male')

plt.show()

sns.barplot(x=korea['EmploymentStatus'].unique(), y=korea['EmploymentStatus'].value_counts())
plt.xticks(rotation=30, ha='right')
plt.title('Employment status of the korean')
plt.ylabel('')
plt.show()

sns.countplot(y='LanguageRecommendationSelect', data=mcq)
plt.show()

sns.countplot(y=mcq['CurrentJobTitleSelect'])
plt.show()

mcq[mcq['CurrentJobTitleSelect'].notnull()]['CurrentJobTitleSelect'].shape

data = mcq[(mcq['CurrentJobTitleSelect'].notnull()) & ((mcq['LanguageRecommendationSelect'] == 'Python') | (mcq['LanguageRecommendationSelect'] == 'R'))]
print(data.shape)
plt.figure(figsize=(8, 10))
sns.countplot(y='CurrentJobTitleSelect', hue='LanguageRecommendationSelect', data=data)
plt.show()

mcq_ml_tool_count = pd.DataFrame(mcq['MLToolNextYearSelect'].value_counts())
mcq_ml_tool_percent = pd.DataFrame(mcq['MLToolNextYearSelect'].value_counts(normalize=True))
mcq_ml_tool_df = mcq_ml_tool_count.merge(mcq_ml_tool_percent, left_index=True, right_index=True).head(20)
mcq_ml_tool_df.columns = ['응답 수', '비율']
mcq_ml_tool_df

data = mcq['MLToolNextYearSelect'].value_counts().head(20)
sns.barplot(y=data.index, x=data)
plt.show()

data = mcq['MLMethodNextYearSelect'].value_counts().head(20)
sns.barplot(y=data.index, x=data)
plt.show()

mcq['LearningPlatformSelect'] = mcq['LearningPlatformSelect'].astype('str')
s = mcq.apply(lambda x: pd.Series(x['LearningPlatformSelect']), axis=1).stack().reset_index(level=1, drop=True)
s.name = 'platform'

plt.figure(figsize=(6,8))
data = s[s != 'nan'].value_counts().head(15)
sns.barplot(y=data.index, x=data)
plt.show()

qc = question.loc[question['Column'].str.contains('LearningCategory')]
print(qc.shape)
qc

use_features = [x for x in mcq.columns if x.find('LearningPlatformUsefulness') != -1]

fdf = {}
for feature in use_features:
    a = mcq[feature].value_counts()
    a = a/a.sum()
    fdf[feature[len('LearningPlatformUsefulness'):]] = a

fdf = pd.DataFrame(fdf).transpose().sort_values('Very useful', ascending=False)

plt.figure(figsize=(10,10))
sns.heatmap(fdf.sort_values("Very useful", ascending=False), annot=True)
plt.show()

fdf.plot(kind='bar', figsize=(20,8), title="Usefulness of Learning Platforms")
plt.show()

cat_features = [x for x in mcq.columns if x.find('LearningCategory') != -1]
cat_features

cdf = {}
for feature in cat_features:
    cdf[feature[len('LearningCategory'):]] = mcq[feature].mean()

cdf = pd.Series(cdf)
cdf

plt.pie(cdf, labels=cdf.index, autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.title("Contribution of each Platform to Learning")
plt.show()

qc = question.loc[question['Column'].str.contains('HardwarePersonalProjectsSelect')]
print(qc.shape)
qc

mcq[mcq['HardwarePersonalProjectsSelect'].notnull()]['HardwarePersonalProjectsSelect'].shape
mcq['HardwarePersonalProjectsSelect'] = mcq['HardwarePersonalProjectsSelect'].astype('str').apply(lambda x: x.split(','))

s = mcq.apply(lambda x: pd.Series(x['HardwarePersonalProjectsSelect']), axis=1).stack().reset_index(level=1, drop=True)
s.name = 'hardware'
s = s[s != 'nan']
pd.DataFrame(s.value_counts())

plt.figure(figsize=(6,8))
sns.countplot(y='TimeSpentStudying', data=mcq, hue='EmploymentStatus').legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

# figure, (ax1, ax2) = plt.subplots(ncols=2)
# figure.set_size_inches(12, 5)
# sns.countplot(x='TimeSpentStudying', data=full_time, hue='EmploymentStatus', ax=ax1).legend(loc='center right', bbox_to_anchor=(1,0.5))
# sns.countplot(x='TimeSpentStuyding', data=looking_for_job, hue='EmplolymentStatus', ax=ax2).legend(loc='center right', bbox_to_anchor=(1,0.5))
# plt.show()

