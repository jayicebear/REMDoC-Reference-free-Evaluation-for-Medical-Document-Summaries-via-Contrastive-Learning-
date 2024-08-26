#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np 
import pandas as pd 
from datasets import load_dataset
import torch

# bio literature review 데이터셋 로드
dataset = load_dataset('allenai/mslr2022', 'cochrane')

target = []
for i in range(len(dataset['train'])):
    target.append(dataset['train'][i]['target'])


# In[4]:





# In[5]:


len(target)


# In[6]:


target[1]


# In[7]:


from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification

tokenizer = AutoTokenizer.from_pretrained("d4data/biomedical-ner-all")
model = AutoModelForTokenClassification.from_pretrained("d4data/biomedical-ner-all")

pipe = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple", device=0)


# In[8]:


ner_list = []
for i in range(len(target)):
    ner_list.append(pipe(target[i]))


# In[9]:


ner_list


# In[10]:


ner_statistic = {}
for i in range(len(ner_list)):
    for j in range(len(ner_list[i])):
        if ner_list[i][j]['entity_group'] in ner_statistic:
            ner_statistic[ner_list[i][j]['entity_group']] += 1
        else:
            ner_statistic[ner_list[i][j]['entity_group']] = 1


# In[11]:


ner_statistic


# In[12]:


import pandas as pd

# Convert the dictionary to a DataFrame for sorting
df = pd.DataFrame(list(ner_statistic.items()), columns=['Entity', 'Count'])

# Sort the DataFrame by 'Count' in descending order
sorted_df = df.sort_values(by='Count', ascending=False)

sorted_df


# In[13]:


Lab_value = []
Detailed_description = []
Therapeutic_procedure = []
Disease_disorder = []
Medication =[]
Diagnostic_procedure = []
Sign_symptom = []
for i in range(len(ner_list)):
    for j in range(len(ner_list[i])):
        if ner_list[i][j]['entity_group'] == 'Lab_value':
            Lab_value.append(ner_list[i][j]['word'])
        if ner_list[i][j]['entity_group'] == 'Detailed_description':
            Detailed_description.append(ner_list[i][j]['word'])   
        if ner_list[i][j]['entity_group'] == 'Therapeutic_procedure':
            Therapeutic_procedure.append(ner_list[i][j]['word'])   
        if ner_list[i][j]['entity_group'] == 'Disease_disorder':
            Disease_disorder.append(ner_list[i][j]['word'])    
        if ner_list[i][j]['entity_group'] == 'Medication':
            Medication.append(ner_list[i][j]['word'])     
        if ner_list[i][j]['entity_group'] == 'Diagnostic_procedure':
            Diagnostic_procedure.append(ner_list[i][j]['word'])   
        if ner_list[i][j]['entity_group'] == 'Sign_symptom':
            Sign_symptom.append(ner_list[i][j]['word'])   


# In[14]:


Lab_value


# In[15]:


for i in range(len(Lab_value)):
    new_text = Lab_value[i].replace('#','')
    Lab_value[i] = new_text
for i in range(len(Detailed_description)):
    new_text = Detailed_description[i].replace('#','')
    Detailed_description[i] = new_text
for i in range(len(Therapeutic_procedure)):
    new_text = Therapeutic_procedure[i].replace('#','')
    Therapeutic_procedure[i] = new_text
for i in range(len(Disease_disorder)):
    new_text = Disease_disorder[i].replace('#','')
    Disease_disorder[i] = new_text
for i in range(len(Medication)):
    new_text = Medication[i].replace('#','')
    Medication[i] = new_text
for i in range(len(Diagnostic_procedure)):
    new_text = Diagnostic_procedure[i].replace('#','')
    Diagnostic_procedure[i] = new_text
for i in range(len(Sign_symptom)):
    new_text = Sign_symptom[i].replace('#','')
    Sign_symptom[i] = new_text


# In[16]:


import random 
random.shuffle(Lab_value)
random.shuffle(Detailed_description)
random.shuffle(Therapeutic_procedure)
random.shuffle(Disease_disorder)
random.shuffle(Medication)
random.shuffle(Diagnostic_procedure)
random.shuffle(Sign_symptom)


# In[17]:


one = Medication[:200]
three = Medication[:200] + Detailed_description[:200] + Disease_disorder[:200]
five = Medication[:200] + Detailed_description[:200] + Disease_disorder[:200] + Therapeutic_procedure[:200] + Sign_symptom[:200]
seven = Medication[:200] + Detailed_description[:200] + Disease_disorder[:200] + Therapeutic_procedure[:200] + Sign_symptom[:200] + Diagnostic_procedure[:200] + Lab_value[:200]


# In[18]:


One = one.copy()
Three = three.copy()
Five = five.copy()
Seven = seven.copy()


# In[19]:


One = One[:500]
three = three[:1000]
five = five[:1500]
seven = seven[:2000]


# In[20]:


one_swap= []
for i,content in enumerate(target):
    changed_content = content
    for j in one:
        if j in One:
            changed_content = changed_content.replace(j,One[-1])
            One.pop() 
    one_swap.append(changed_content)


# In[21]:


# print(len(target))
print(len(one) * len(target))


# In[22]:


three_swap= []
for i,content in enumerate(target):
    changed_content = content
    for j in three:
        if j in content and Three:
            changed_content = changed_content.replace(j,Three[-1])
            Three.pop() 
    three_swap.append(changed_content)


# In[23]:


five_swap= []
for i,content in enumerate(target):
    changed_content = content
    for j in five:
        if j in content and Five:
            changed_content = changed_content.replace(j,Five[-1])
            Five.pop() 
    five_swap.append(changed_content)


# In[24]:


seven_swap= []
for i,content in enumerate(target):
    changed_content = content
    for j in seven:
        if j in content and Seven:
            changed_content = changed_content.replace(j,Seven[-1])
            Seven.pop() 
    seven_swap.append(changed_content)


# In[25]:


import json
with open('one_shuffled.json', 'w') as json_file:
    json.dump(one_swap, json_file)


# In[26]:


import json
with open('Three_shuffled.json', 'w') as json_file:
    json.dump(three_swap, json_file)


# In[27]:


import json
with open('Five_shuffled.json', 'w') as json_file:
    json.dump(five_swap, json_file)


# In[28]:


import json
with open('Seven_shuffled.json', 'w') as json_file:
    json.dump(seven_swap, json_file)


# In[ ]:


Medication	7855
1	Detailed_description	4750
5	Disease_disorder	4011
2	Therapeutic_procedure	3063
11	Sign_symptom	2935
7	Diagnostic_procedure	2661
0	Lab_value	1554
6	Coreference	1489
10	Biological_structure	1260


# In[2]:


import matplotlib.pyplot as plt

# 주어진 데이터
data = {
    'Lab_value': 1554,
    'Detailed_description': 4750,
    'Therapeutic_procedure': 3063,
    'Disease_disorder': 4011,
    'Coreference': 1489,
    'Diagnostic_procedure': 2661,
    'Medication': 7855,
    'Coreference' : 1489,
    'Biological_structure' : 1260
}

sorted_data = dict(sorted(data.items(), key=lambda item: item[1], reverse=True))

# 데이터 분리
labels = list(sorted_data.keys())
sizes = list(sorted_data.values())

# 원형 차트 그리기
plt.figure(figsize=(12, 12))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140,counterclock=False)
plt.axis('equal')  # 원형으로 그리기 위해 비율을 같게 조정

# 제목 추가
#plt.title('Distribution of Medical Categories (Sorted by Percentage)')

# 차트 표시
plt.show()


# In[1]:


import matplotlib.pyplot as plt

# 주어진 데이터
data = {
    'Lab_value': 1554,
    'Detailed_description': 4750,
    'Therapeutic_procedure': 3063,
    'Disease_disorder': 4011,
    'Coreference': 1489,
    'Diagnostic_procedure': 2661,
    'Medication': 7855
}

# 중복된 'Coreference' 제거 후 정렬
sorted_data = dict(sorted(data.items(), key=lambda item: item[1], reverse=True))

# 데이터 분리
labels = list(sorted_data.keys())
sizes = list(sorted_data.values())

# 원형 차트 그리기
plt.figure(figsize=(12, 12))
wedges, texts, autotexts = plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, counterclock=False, textprops=dict(color="w"))

# 원형으로 그리기 위해 비율을 같게 조정
plt.axis('equal')

# 범례 추가
plt.legend(wedges, labels, title="Medical Categories", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

# 제목 추가
# plt.title('Distribution of Medical Categories (Sorted by Percentage)')

# 차트 표시
plt.show()

