from torch.utils.data import Dataset
import torch.nn as nn
from transformers import RobertaModel
import torch
import torch.nn.functional as F

class MedicalSummaryDataset(Dataset):
    def __init__(self, tokenizer, data, max_length=512):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        doc = item["document"]
        pos = item["positive_summary"]
        negs = item["negative_summaries"]  # list of strings

        doc_enc = self.tokenizer(doc, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
        pos_enc = self.tokenizer(pos, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
        neg_encs = [self.tokenizer(n, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt") for n in negs]

        return {
            "doc_input": doc_enc,
            "pos_input": pos_enc,
            "neg_inputs": neg_encs
        }


class DualEncoder(nn.Module):
    def __init__(self, model_name="roberta-large"):
        super().__init__()
        self.doc_encoder = RobertaModel.from_pretrained(model_name)
        self.sum_encoder = RobertaModel.from_pretrained(model_name)
        self.pooling = lambda x: x.last_hidden_state[:, 0]  # [CLS] token

    def forward(self, doc_input, sum_input):
        doc_vec = self.pooling(self.doc_encoder(**doc_input))
        sum_vec = self.pooling(self.sum_encoder(**sum_input))
        return doc_vec, sum_vec

def multi_positive_info_nce_loss(doc_vec, pos_vecs, neg_vecs, temperature=0.05):
    losses = []
    for pos_vec in pos_vecs:
        all_vecs = torch.cat([pos_vec] + neg_vecs, dim=0)  # (1 + N, dim)
        logits = F.cosine_similarity(doc_vec, all_vecs) / temperature
        logits = logits.unsqueeze(0)  # (1, N+1)
        labels = torch.tensor([0]).to(doc_vec.device)
        loss = F.cross_entropy(logits, labels)
        losses.append(loss)
    return torch.stack(losses).mean()

model = DualEncoder().to("cuda")
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

model.train()
for epoch in range(epochs):
    for batch in tqdm(dataloader):
        doc_input = {k: v.squeeze(0).to("cuda") for k, v in batch["doc_input"].items()}
        pos_input = {k: v.squeeze(0).to("cuda") for k, v in batch["pos_input"].items()}
        neg_inputs = [ {k: v.squeeze(0).to("cuda") for k, v in neg.items()} for neg in batch["neg_inputs"]]

        doc_vec, pos_vec = model(doc_input, pos_input)
        neg_vecs = [model(None, neg_input)[1] for neg_input in neg_inputs]  # only summary encoding

        loss = info_nce_loss(doc_vec, pos_vec, neg_vecs)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
