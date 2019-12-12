# FROM https://github.com/chadwgardner/kaggle_survey/blob/master/.ipynb_checkpoints/kaggle-checkpoint.ipynb

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

sns.set()

def select_all_that_apply_plot(df, question, figsize=(12,36)):
    filtered = pd.DataFrame(df.loc[:,df.columns.str.startswith(question)])
    split = filtered[question].dropna().str.split(',').tolist()
    flattened = []
    for i in split:
        for j in i:
            flattened.append(j)
    flattened_DF = pd.DataFrame(flattened, columns=[question])
    plt.figure(figsize=(12, 6))

    ax = sns.countplot(y=question, data=flattened_DF, order=flattened_DF[question].value_counts().index);
    plt.setp(ax.get_yticklabels(), fontsize=15)
    plt.ylabel('');
    plt.title(question + ', N = ' + str(len(filtered)))
    plt.show()
    return

def multi_plot_hist(df, question_stem, figsize=(24,18)):
    filtered = pd.DataFrame(df.loc[:,df.columns.str.startswith(question_stem)]).dropna()
    num_columns = len(filtered.columns)
    plt.figure(figsize=figsize)
    for i in range(num_columns):
        plt.subplot(math.ceil(num_columns/3),3,i+1)
        plt.title(filtered.columns[i][len(question_stem):])
        plt.xlabel('Percentage')
        plt.hist(filtered[filtered.columns[i]], rwidth=0.8)
    plt.show()
    return filtered

def replace_usefulness(df, question_stem):
    filtered = pd.DataFrame(df.loc[:,df.columns.str.startswith(question_stem)])
    filtered.replace({'Very useful': 1, 'SomeWhat useful': 0.5, 'Not Useful': 0, np.nan:0}, inplace=True)
    return filtered

### What errors?
def plot_usefulness_questions(df, question_stem, figsize=(12,36), drop_last=None):

    #Use our replace_usefuless function
    replaced = replace_usefulness(df, question_stem)
    
    #Add up all the values, sort them
    normed = replaced.sum().sort_values(ascending=False)
   
    #Remove the question stem from each row index, leaving only the unique sub-question text
    normed.index=[s[len(question_stem):] for s in normed.index]
    
    #Drop some of the last ones if needed
    if drop_last != None: 
        normed.drop(normed.index[-1*drop_last:], inplace=True)
    
    plt.figure(figsize=figsize)
    ax = sns.barplot(y = normed.index, x = normed)
    plt.setp(ax.get_yticklabels(), fontsize=15)
    plt.title(question_stem + ', N = ' + str(len(replaced)))
    plt.xlabel('Usefulness')
    plt.show()
    
    return normed

def replace_frequency(df, question_stem):
    filtered = pd.DataFrame(df.loc[:,df.columns.str.startswith(question_stem)])
    filtered.replace({'Most of the time': 1, 'Often': 0.6, 'Sometimes': 0.26, 'Rarely': 0.1}, inplace=True)
    return filtered

def plot_frequency_questions(df, question_stem, figsize=(12,36), drop_last=None):
    replaced = replace_frequency(df, question_stem)
    normed = replaced.sum().sort_values(ascending=False)
    normed.index = [s[len(question_stem):] for s in normed.index]
    if drop_last != None:
        normed.drop(normed.index[-1*drop_last:], inplace=True)
    plt.figure(figsize=figsize)
    ax = sns.barplot(y=normed.index, x=normed)
    plt.setp(ax.get_yticklabels(), fontsize=15)
    plt.title(question_stem + ', N = ' + str(len(replaced)))
    plt.xlabel('Raw Score')
    plt.show()
    return normed

def replace_importance(df, question_stem):
    filtered = pd.DataFrame(df.loc[:,df.columns.str.startswith(question_stem)])
    if 'Necessary' in filtered.values:
        replacements = {'Necessary': 1, 'Nice to have': 0.5, 'Unnecessary': 0, np.nan: 0}
    else:
        replacements = {'Very Important': 1, 'Somewhat important': 0.5, 'Not important': 0, np.nan: 0}
    filtered.replace(replacements, inplace=True)
    return filtered

def plot_importance_questions(df, question_stem, figsize=(12,36), drop_last=None):
    replaced = replace_importance(df, question_stem)
    normed = replaced.sum().sort_values(ascending=False)
    normed.index = [s[len(question_stem):] for s in normed.index]
    if drop_last != None:
        normed.drop(normed.index[-1*drop_last:], inplace=True)
    plt.figure(figsize=figsize)
    ax = sns.barplot(y=normed.index, x=normed)
    plt.setp(ax.get_yticklabels(), fontsize=15)
    plt.title(question_stem + ', N = ' + str(len(replaced)))
    plt.xlabel('Importance')
    plt.show()
    return normed

MC = pd.read_csv('ds_survey_2017/multipleChoiceResponses.csv', encoding='latin-1', low_memory=False)

MC.Age.hist(bins=20, figsize=(12,6), rwidth=0.85)
plt.axvline(x=MC.Age.median(), color='black', linestyle='--', label='Median: ' + str(MC.Age.median()) + ' years')
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Age of Survey Respondents N = ' + str(MC.Age.count()))
plt.legend(loc='upper right')
plt.show()

fraction_under_40 = MC.Age[MC.Age < 40].count()/ len(MC.Age)
print("Fraction under 40 years old = " + str(fraction_under_40))

money = MC.CompensationAmount[MC.CompensationCurrency == 'USD']

money = money.str.replace(',', '')
money.dropna(inplace=True)
money = pd.to_numeric(money, errors='coerce')
money.sort_values(inplace=True)
money.drop([3013,5939], inplace=True)

money_median = money.median()
money_less = money[money <= 300000]

money_less.hist(bins=30, histtype='bar', figsize=(12, 6), rwidth=0.85)
plt.axvline(x=money_median, linestyle='--', color='darkred', label='Median = $' + str(money_median))
plt.xlabel('Salary in USD')
plt.ylabel('Count')
plt.title('Compensation in USD, N = ' + str(len(money)))
plt.legend(loc='upper right')
plt.show()

plt.figure(figsize=(12,6))
plt.title('Tenure Writing Data Code, N = ' + str(MC.Tenure.count()))
sns.countplot(data=MC, y='Tenure', order=['Less than a year', '1 to 2 years', '3 to 5 years', '6 to 10 years', 'More than 10 years', 'I don\'t write code to analyze data'])
plt.ylabel('')
plt.show()

replacements = {"I don't write code to analyze data": np.nan, "1 to 2 years": 2, "3 to 5 years": 4, "6 to 10 years": 8, "Less than a year": 0.5, "More than 10 years": 11}
x_tenure = MC.Tenure.replace(replacements)[money_less.index].dropna()
y_money = money_less[x_tenure.index]
plt.figure(figsize=(12,6))
sns.boxplot(x_tenure, y_money)
plt.ylim(0, 400000)
plt.title('Compensation vs. Tenure, N = ' + str(len(money)))
plt.show()

plt.figure(figsize=(12,6))
ax = sns.countplot(y='FormalEducation', data=MC, order=MC['FormalEducation'].value_counts().index)
plt.setp(ax.get_yticklabels(), fontsize=15)
plt.ylabel('')
plt.title('Level of Formal Education, N = ' + str(MC.FormalEducation.count()))
plt.show()

fraction_higher_bachelors = (MC['FormalEducation'].value_counts()['Master\'s degree']+MC['FormalEducation'].value_counts()['Doctoral degree'])/len(MC.FormalEducation)
print('Fraction of respondents with more than a Bachelor\'s degree: ' + str(fraction_higher_bachelors))

plt.figure(figsize=(12,6))
plt.title('First Data Science Training, N = ' + str(MC.FirstTrainingSelect.count()))
ax = sns.countplot(y='FirstTrainingSelect', data=MC, order=MC['FirstTrainingSelect'].value_counts().index)
plt.setp(ax.get_yticklabels(), fontsize=15)
plt.ylabel('')
plt.show()

# plot_usefulness_questions

sns.set_palette(sns.cubehelix_palette(20, start=1.7, reverse=True))
# plot_usefulness_questions(MC, 'LearningPlatformUsefulness', figsize=(10,10))

# multi_plot_hist

filt = multi_plot_hist(MC, 'LearningCategory', figsize=(20,12))

plt.figure(figsize=(12,6))
plt.title('Most important way to prove knowledge, N = ' + str(MC.ProveKnowledgeSelect.count()))
ax = sns.countplot(data=MC, y='ProveKnowledgeSelect', order=MC['ProveKnowledgeSelect'].value_counts().index)
plt.setp(ax.get_yticklabels(), fontsize=15)
plt.ylabel('')
plt.show()

sns.set_palette(sns.cubehelix_palette(20, start=1.7, reverse=True))
plt.figure(figsize=(14,6))
plt.title('Industry, N = ' + str(len(MC.EmployerIndustry)))
ax = sns.countplot(y='EmployerIndustry', data=MC, order=MC['EmployerIndustry'].value_counts().index)

plt.setp(ax.get_yticklabels(), fontsize=13)
plt.ylabel('')
plt.show()

plt.figure(figsize=(8,8))
plt.title('Job Function')
ax = sns.countplot(y='JobFunctionSelect', data=MC, order=MC['JobFunctionSelect'].value_counts().index)
plt.setp(ax.get_yticklabels(), fontsize=13)
plt.ylabel('')
plt.show()

sns.set_palette(sns.cubehelix_palette(30, start=1.7, reverse=True))
plot_frequency_questions(MC, 'WorkToolsFrequency', figsize=(10,8), drop_last=25)

sns.set_palette(sns.cubehelix_palette(35, start=1.7, reverse=True))
plot_frequency_questions(MC, 'WorkMethodsFrequency', figsize=(10,8), drop_last = 12)

df = plot_importance_questions(MC, 'JobSkillImportance', figsize=(10,7), drop_last=3)

plt.figure(figsize=(12,6))
plt.title('Recommended Language')
sns.set_palette(sns.cubehelix_palette(13, start=1.7, reverse=True))
ax = sns.countplot(data=MC, y='LanguageRecommendationSelect', order=MC['LanguageRecommendationSelect'].value_counts().index);
plt.ylabel('')
plt.setp(ax.get_yticklabels(), fontsize=14)
plt.show()

plt.figure(figsize=(12,6))
plt.title('Job Search')
ax = sns.countplot(y='JobSearchResource', data=MC, order=MC['JobSearchResource'].value_counts().index)
plt.setp(ax.get_yticklabels(), fontsize=15)
plt.ylabel('')
plt.show()


sns.set_palette(sns.cubehelix_palette(13, start=1.7, reverse=True))
select_all_that_apply_plot(MC, 'MLSkillsSelect', figsize=(10,8))


plt.figure(figsize=(10,6))
plt.title('Level of Algorithum Understanding')
ax = sns.countplot(y='AlgorithmUnderstandingLevel', data=MC, order=MC['AlgorithmUnderstandingLevel'].value_counts().index);
plt.setp(ax.get_yticklabels(), fontsize=13)
plt.ylabel('')
plt.show()

sns.set_palette(sns.cubehelix_palette(30, start=1.7, reverse=True))
plot_frequency_questions(MC, 'WorkChallengeFrequency', figsize=(12,8), drop_last=6)

sns.set_palette(sns.cubehelix_palette(20, start=1.7, reverse=True))
plot_importance_questions(MC, 'JobFactor', figsize=(12,8))


plt.figure(figsize=(8,6))
plt.title('Data Scientist')
ax = sns.countplot(y='DataScienceIdentitySelect', data=MC, order=MC['DataScienceIdentitySelect'].value_counts().index);
plt.setp(ax.get_yticklabels(), fontsize=15)
plt.ylabel('')
plt.show()

plt.figure(figsize=(14,6))
sns.set_palette(sns.cubehelix_palette(12, start=1.7, reverse=True))
plt.title('Machine Learning at Work, N = ' + str(MC.EmployerMLTime.count()))
ax = sns.countplot(y='EmployerMLTime', data=MC, order=['Less than one year', '1-2 years','3-5 years','6-10 years','More than 10 years','Don\'t know']);
plt.ylabel('')
plt.setp(ax.get_yticklabels(), fontsize=15)
plt.show()


plt.figure(figsize=(14,7))
plt.title('Job Satisfaction, N = ' + str(MC.JobSatisfaction.count()))
plt.ylabel('')
sns.countplot(x='JobSatisfaction', data=MC, order=['1 - Highly Dissatisfied','2', '3','4','5','6','7','8','9','10 - Highly Satisfied',])
plt.show()
