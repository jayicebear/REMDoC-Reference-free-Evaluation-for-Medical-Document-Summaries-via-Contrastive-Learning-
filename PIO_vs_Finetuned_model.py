#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Load data

import os
import csv
import json
import simpledorff
from collections import defaultdict, Counter
from pprint import pprint
import numpy as np
import itertools
import random
import editdistance

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, kendalltau

DATA_FILE = '../Correlation_Test_data & Model/mslr-annotated-dataset/data/data_with_overlap_scores.json'
SUBMISSION_FILE = '../Correlation_Test_data & Model/mslr-annotated-dataset/data/submission_info.json'
PAIRWISE_DIR = '../Correlation_Test_data & Model/mslr-annotated-dataset/data/Pairwise/annotated_samples/'

data = []
with open(DATA_FILE, 'r') as f:
    for line in f:
        data.append(json.loads(line))

with open(SUBMISSION_FILE, 'r') as f:
    submissions = json.load(f)


# In[3]:


# Shorthand names for systems included in analysis

NAME_MAP = {
    'led-base-16384-cochrane': 'LED-base-16k',
    'SciSpace': 'SciSpace',
    'ittc2': 'ITTC-2',
    'ittc1': 'ITTC-1',
    'bart-large-finetuned': 'BART-large',  
    'AI2/Longformer BART/train MS2/decode Cochrane': 'LED-MS2',
    'AI2/Longformer BART/Train Cochrane/Decode Cochrane': 'LED-Cochrane',
    'AI2/BART/train Cochrane/decode Cochrane': 'BART-Cochrane',
    'longt5_pubmed': 'LongT5-PubMed'
}


# In[12]:


# Report dataset test set size and number of entries with annotations

cochrane = [d for d in data if d['subtask'] == 'Cochrane']
ms2 = [d for d in data if d['subtask'] == 'MS2']

cochrane_annot = [d for d in cochrane if any(entry['annotations'] for entry in d['predictions'])]
ms2_annot = [d for d in ms2 if any(entry['annotations'] for entry in d['predictions'])]

cochrane_metrics = [d for d in cochrane if any(entry['scores'] for entry in d['predictions'])]
ms2_metrics = [d for d in ms2 if any(entry['scores'] for entry in d['predictions'])]

print('Cochrane: ', len(cochrane))
print('w/ annotations: ', len(cochrane_annot))
print('w/ metrics: ', len(cochrane_metrics))
print()


# In[13]:


cochrane


# In[5]:


# Keeps Cochrane test examples
# Add additional fields corresponding to annotated facets

data_to_analyze = []
for entry in cochrane:
    for pred in entry['predictions']:
        data_entry = {
            'review_id': entry['review_id'],
            'exp_id': pred['exp_short'],
            'exp_name': submissions[pred['exp_short']]['name'],
            'exp_short': NAME_MAP.get(submissions[pred['exp_short']]['name'], None)
        }
        annot_dict = dict()
        annot_dict['annotated'] = False
        if pred['annotations']:
            annot = pred['annotations'][0]
            if not annot['population']:
                annot['population'] = 0
            if not annot['intervention']:
                annot['intervention'] = 0
            if not annot['outcome']:
                annot['outcome'] = 0
            if not annot['ed_target']:
                annot['ed_target'] = -1
            if not annot['ed_generated']:
                annot['ed_generated'] = -1
            if not annot['strength_target']:
                annot['strength_target'] = 0
            if not annot['strength_generated']:
                annot['strength_generated'] = 0
            annot_dict = {
                'annot_id': annot['annot_id'],
                'Fluency': annot['fluency'] / 2.0,
                'PIO': (annot['population'] + annot['intervention'] + annot['outcome']) / 6.0,
                # new normalized ED (just check if agree)
                'Direction': int(annot['ed_target'] == annot['ed_generated']),
                # old normalized ED (positive-no effect-negative)
                'Direction_old': (2.0 - abs(abs(annot.get('ed_target')) - abs(annot.get('ed_generated')))) / 2.0,
                'Strength': (3.0 - (abs(annot.get('strength_target')) - abs(annot.get('strength_generated')))) / 3.0
            }
            annot_dict['annot_all'] = np.mean([
                annot_dict['Fluency'], 
                annot_dict['PIO'], 
                annot_dict['Direction'], 
                annot_dict['Strength']
            ])
            annot_dict['annotated'] = True
        
        # metric dict
        score_dict = dict()
        score_dict['scored'] = False
        if pred['scores']:
            score_dict = {
                'ROUGE-1': pred['scores']['rouge1_f'],
                'ROUGE-2': pred['scores']['rouge2_f'],
                'ROUGE-L': pred['scores']['rougeL_f'],
                'Avg-ROUGE-F': np.mean([
                    pred['scores']['rouge1_f'],
                    pred['scores']['rouge2_f'],
                    pred['scores']['rougeL_f']
                ]),
                'ROUGE-L-Sum': pred['scores']['rougeLsum_f'],
                'BERTScore-F': pred['scores']['bertscore_f'] if pred['scores']['bertscore_f'] > 0.5 else None,
                'Delta-EI': pred['scores']['ei_score'],
                'ClaimVer': pred['scores']['claimver'],
                'STS': pred['scores']['sts'],
                'NLI': pred['scores']['nli']
            }
            score_dict['scored'] = True
            
        # compute PIO overlap scores
        overlap_scores_dict = dict()
        overlap_scores_dict['overlap_scored'] = False
        if pred['overlap_scores']:
            overlap_scores_dict = {
                'p_overlap_exact_match': pred['overlap_scores']['exact_match']['PAR'],
                'i_overlap_exact_match': pred['overlap_scores']['exact_match']['INT'],
                'o_overlap_exact_match': pred['overlap_scores']['exact_match']['OUT'],

                'p_overlap_close_match': pred['overlap_scores']['close_match']['PAR'],
                'i_overlap_close_match': pred['overlap_scores']['close_match']['INT'],
                'o_overlap_close_match': pred['overlap_scores']['close_match']['OUT'],

                'p_overlap_substring': pred['overlap_scores']['substring']['PAR'],
                'i_overlap_substring': pred['overlap_scores']['substring']['INT'],
                'o_overlap_substring': pred['overlap_scores']['substring']['OUT'],

                'PIO-Overlap': np.mean([
                    pred['overlap_scores']['substring']['PAR'],
                    pred['overlap_scores']['substring']['INT'],
                    pred['overlap_scores']['substring']['OUT'],
                ])
            }
            overlap_scores_dict['overlap_scored'] = True
        data_entry.update(annot_dict)
        data_entry.update(score_dict)
        data_entry.update(overlap_scores_dict)
        data_to_analyze.append(data_entry)


# In[41]:


len(data_to_analyze)


# In[6]:


pio_lst = []
fluency = []
strength = []
direction = []
rouge = []
DeltaEI = []
ClaimVer = []
BERTScore = []
sts = []
nli = []
err = []
for i in range(len(data_to_analyze)):
    try:
        pio_lst.append(data_to_analyze[i]['PIO'])
        fluency.append(data_to_analyze[i]['Fluency'])
        strength.append(data_to_analyze[i]['Strength'])
        direction.append(data_to_analyze[i]['Direction'])
        rouge.append(data_to_analyze[i]['ROUGE-1'])
        DeltaEI.append(data_to_analyze[i]['Delta-EI'])
        ClaimVer.append(data_to_analyze[i]['ClaimVer'])
        BERTScore.append(data_to_analyze[i]['BERTScore-F'])
        sts.append(data_to_analyze[i]['STS'])
        nli.append(data_to_analyze[i]['NLI'])
    except:
        err.append(i)


# In[7]:


BERTScore = []
sts = []
nli = []
for i in range(len(data_to_analyze)):
    try:
        BERTScore.append(data_to_analyze[i]['BERTScore-F'])
        sts.append(data_to_analyze[i]['STS'])
        nli.append(data_to_analyze[i]['NLI'])
    except:
        print('err')


# In[8]:


len(err)


# In[9]:


exp_set= []
set_list = []
cnt = 0
for i, content in enumerate(cochrane):
    for j in range(len(cochrane[i]['predictions'])):
        if cnt not in err:
            set_list.append((cochrane[i]['target'],cochrane[i]['predictions'][j]['prediction']))
            exp_set.append(cochrane[i]['predictions'][j]['exp_short'])
        cnt += 1


# In[10]:


print(len(DeltaEI))
print(len(BERTScore))
print(len(sts))
print(len(nli))
print(len(ClaimVer))
print(len(set_list))
print(len(ClaimVer))
print(len(data_to_analyze))


# In[11]:


set_list


# In[12]:


import json
with open('Test_data/set_list.json','w') as f:
    json.dump(set_list,f)
# import json
# with open('set_list.json','w') as f:
#     json.dump(set_list,f)

import json
with open('Test_data/pio_lst.json','w') as f:
    json.dump(pio_lst,f)
    

# import json
# with open('set_list.json','w') as f:
#     json.dump(set_list,f)

# import json
# with open('pio_lst.json','w') as f:
#     json.dump(pio_lst,f)


# In[24]:


rouge_score_list = []
for i in range(len(rouge_score)):
    rouge_score[0][0]['rouge-1']['r']


# In[12]:


from rouge import Rouge
rouge = Rouge()

rouge_score = []
for i in range(len(set_list)):
    rouge_score.append(rouge.get_scores(set_list[i][0], set_list[i][1]))


# In[16]:


set_list[300]


# In[12]:


from sentence_transformers import SentenceTransformer, util
import torch
model = torch.load('Test_models/final_roberta_large.pth')
model_score = []

for i in range(len(set_list)):
    embedding1 = model.encode(set_list[i][0], convert_to_tensor=True)
    embedding2 = model.encode(set_list[i][1], convert_to_tensor=True)
    cosine_similarity = util.pytorch_cos_sim(embedding1, embedding2)
    model_score.append(cosine_similarity.item())
#print("Cosine Similarity:", cosine_similarity.item())


# In[13]:


correlation = np.corrcoef(model_score, strength)[0, 1]

print("상관계수:", correlation)  


# In[23]:


len(model_score)


# In[51]:


df = pd.DataFrame({
    'PIO': pio_lst,
    'Rouge': rouge,
    'DeltaEI': DeltaEI,
    'ClaimVer': ClaimVer,
    'BERTScore': BERTScore,
    'STS': sts,
    'NLI': nli,
    'Ours': model_score,
})


# In[52]:


df


# In[53]:


correlation_matrix = df.corr()

# Step 3: Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix Heatmap')
plt.show()


# In[54]:


import numpy as np

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

# Set up the matplotlib figure
plt.figure(figsize=(10, 8))

# Draw the heatmap with the mask
sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1, square=True, linewidths=.5)

plt.show()


# In[50]:


pdf_path = "./correlation_matrix_heatmap.pdf"
plt.savefig(pdf_path, format='pdf')


# In[7]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# 데이터 배열 생성
data = np.array([
    [1, -0.027, -0.08, 0.14, 0.022, 0.066, 0.053, 0.5],
    [-0.027, 1, -0.15, 0.48, 0.75, 0.56, 0.55, 0.05],
    [-0.08, -0.15, 1, -0.089, -0.21, -0.29, -0.3, -0.0027],
    [0.14, 0.48, -0.089, 1, 0.59, 0.59, 0.5, 0.26],
    [0.022, 0.75, -0.21, 0.59, 1, 0.63, 0.56, 0.1],
    [0.066, 0.56, -0.29, 0.59, 0.63, 1, 0.93, 0.14],
    [0.053, 0.55, -0.3, 0.5, 0.56, 0.93, 1, 0.12],
    [0.5, 0.05, -0.0027, 0.26, 0.1, 0.14, 0.12, 1]
])

# 데이터프레임 생성
columns = ['PIO', 'Rouge', 'DeltaEI', 'ClaimVer', 'BERTScore', 'STS', 'NLI', 'Ours']
df = pd.DataFrame(data, columns=columns, index=columns)

# 상삼각형을 남기고 나머지를 NaN으로 대체
mask = np.triu(np.ones_like(df, dtype=bool), k=1)

# 히트맵 그리기
plt.figure(figsize=(10, 8))
heatmap = sns.heatmap(df, annot=True, mask=mask, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)

# 그래프 보여주기
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


## MOdel Test 


# In[89]:


from sentence_transformers import SentenceTransformer, util
import torch
from scipy import stats
import numpy
from transformers import BartTokenizer, BartModel


model = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO-SCIFACT')
#model = BartTokenizer.from_pretrained('facebook/bart-large')
model_score = []

for i in range(len(set_list)):
    embedding1 = model(set_list[i][0])['input_ids'][0]
    embedding2 = model(set_list[i][1])['input_ids'][0]
    cosine_similarity = util.pytorch_cos_sim(embedding1, embedding2)
   # pearson_similarity = stats.pearsonr(embedding1.cpu().numpy(), embedding2.cpu().numpy())
    model_score.append(cosine_similarity.statistic)
   # model_score.append(pearson_similarity.statistic)


# In[9]:


pio_lst


# In[11]:


import seaborn as sns
sns.histplot(model_score, binwidth=0.1, kde=True)


# In[16]:


from datasets import load_dataset
example = load_dataset('allenai/mslr2022','cochrane')


# In[45]:


print(example['train'][21])


# In[9]:


from rouge_score import rouge_scorer

def compare_rouge(document: str, summary: str):
    # Initialize the ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Calculate ROUGE scores
    scores = scorer.score(document, summary)
    
    # Print the results
    for key, value in scores.items():
        print(f"{key}: Precision: {value.precision:.4f}, Recall: {value.recall:.4f}, F1 Score: {value.fmeasure:.4f}")

# Example usage
document = """The clinical features of 49 patients who had sustained small strokes
in the internal carotid artery territory, who were normotensive, free
from cardiac or other relevant disease, and who each had a normal
appropriate single vessel angiogram are presented. These were
randomized into two groups: group A, 25 patients, who received
only supportive treatment; group B, 24 patients who were treated
with anticoagulants for an average period of 18 months. There was
a reduced incidence of neurological episodes during the administra-
tion of anticoagulant therapy but, after treatment was discontinued,
there was no significant difference between the two groups. In view
of the relatively benign prognosis for this syndrome, unless special
facilities exist for the personal control of anticoagulant treatment,
the dangers may outweigh the benefits."""

summary = """Compared with control, there was no evidence of benefit from
long-term anticoagulant therapy in people with presumed non-
cardioembolic ischaemic stroke or transient ischaemic attack, but
there was a significant bleeding risk."""
summary2 = '''
This study deals with 49 patients who, after having small strokes,
suddenly turned into superheroes. Group A received no treatment at
all, yet they went on to enjoy life as if nothing had happened. Mean-
while, Group B took anticoagulants for 18 months and magically
resolved all their neurological issues, almost as if they had gained
superpowers. However, as soon as they stopped the medication,
they reverted back to normal humans. This study suggests that all
patients have some kind of magical recovery ability.
'''
summary3 = '''
The study focused on 49 patients who liked to wear red socks
after having small strokes. Group A wore red socks for 18 months,
while Group B did not. Surprisingly, those who wore red socks
experienced fewer neurological episodes during the study. However,
when they stopped wearing the socks, there was no significant
difference between the two groups. This study suggests that the
color of your socks might be more important than medication in
stroke recovery, though further research is needed
'''
compare_rouge(document, summary3)


# In[11]:


from bert_score import score

def compare_bertscore(document: str, summary: str, lang='en'):
    # Compute BERTScore
    P, R, F1 = score([summary], [document], lang=lang, verbose=True)

    # Print the results
    print(f"BERTScore Precision: {P.mean():.4f}")
    print(f"BERTScore Recall: {R.mean():.4f}")
    print(f"BERTScore F1 Score: {F1.mean():.4f}")

compare_bertscore(document, summary3)


# In[29]:


from sentence_transformers import SentenceTransformer, util
import torch
model = torch.load('Test_models/final_roberta_large.pth')
model_score = []

for i in range(len(set_list)):
    embedding1 = model.encode(document, convert_to_tensor=True)
    embedding2 = model.encode(summary2, convert_to_tensor=True)
    cosine_similarity = util.pytorch_cos_sim(embedding1, embedding2)
    model_score.append(cosine_similarity.item())


# In[30]:


model_score


# In[18]:


from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

#model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# model_id = "beomi/Llama-3-Open-Ko-8B-Instruct-preview"
# model_id = "beomi/Llama-3-KoEn-8B-Instruct-preview"
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,

    # torch_dtype=torch.bfloat16,
    torch_dtype="auto",
    device_map="auto",
)


# In[19]:


def generate_response(system_message, user_message):
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=512,
        eos_token_id=terminators,
        do_sample=True,
        # temperature=1,
        temperature=0.6,
        top_p=0.9
        
    )

    response = outputs[0][input_ids.shape[-1]:]

    return tokenizer.decode(response, skip_special_tokens=True)


# In[ ]:



llama3_summary_text = generate_response(system_message="너는 질문에 답하는 챗봇이야.document 와 summary를 보고 얼마나 summary 가 잘 되었는지 0에서 1점 사이로 답해줘",
                             user_message=f"document:{document},summary:{summary3}")
print(llama3_summary_text)


# In[ ]:




