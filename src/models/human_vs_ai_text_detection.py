
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score,confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import OneClassSVM

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup,BertModel

import torch
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import uniform

df = pd.read_csv("AI_Human.csv")


nltk.download('stopwords')
nltk.download('wordnet')

english_stopwords = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', ' ', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in english_stopwords])
    return text

df['clean_text'] = df['text'].apply(clean_text)

human_data = df[df['generated'] == 0.0]
human_text = ' '.join(human_data['clean_text'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(human_text)

ai_data = df[df['generated'] == 1.0]
ai_text = ' '.join(ai_data['clean_text'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(ai_text)


vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_text'])
y = df['generated']



ml_df = df.sample(n=10000, random_state=42)

X_ml = vectorizer.fit_transform(ml_df['clean_text'])
y_ml = ml_df['generated']

X_train_ml, X_test_ml, y_train_ml, y_test_ml = train_test_split(X_ml, y_ml, test_size=0.2, random_state=42)


log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_ml, y_train_ml)
y_pred_log_reg = log_reg.predict(X_test_ml)

svm = SVC(kernel='linear')
svm.fit(X_train_ml, y_train_ml)
y_pred_svm = svm.predict(X_test_ml)

nn_df = df.sample(n=250000, random_state=42)

max_words = 10000
max_len = 100

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(nn_df['clean_text'])
sequences = tokenizer.texts_to_sequences(nn_df['clean_text'])
X_nn = pad_sequences(sequences, maxlen=max_len)

label_encoder = LabelEncoder()
y_nn = label_encoder.fit_transform(nn_df['generated'])

X_train_nn, X_test_nn, y_train_nn, y_test_nn = train_test_split(X_nn, y_nn, test_size=0.2, random_state=42)


embedding_dim = 64
model_nn = Sequential([
    Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len),
    LSTM(64, return_sequences=True),
    LSTM(32),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model_nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


history = model_nn.fit(X_train_nn, y_train_nn, epochs=5, batch_size=64, validation_data=(X_test_nn, y_test_nn))

nn_loss, nn_accuracy = model_nn.evaluate(X_test_nn, y_test_nn)

import torch
from GPUtil import showUtilization as gpu_usage
from numba import cuda

def free_gpu_cache():
    print("Initial GPU Usage")
    gpu_usage()

    torch.cuda.empty_cache()

    cuda.select_device(0)
    cuda.close()
    cuda.select_device(0)

    print("GPU Usage after emptying the cache")
    gpu_usage()

free_gpu_cache()


bert_df = df.sample(n=300000, random_state=42)


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

input_ids = []
attention_masks = []

for text in bert_df['clean_text']:
    encoded = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=100,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    input_ids.append(encoded['input_ids'])
    attention_masks.append(encoded['attention_mask'])

input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(bert_df['generated'].values.astype(int))

dataset = TensorDataset(input_ids, attention_masks, labels)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=32)
test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=32)

model_bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)


optimizer = AdamW(model_bert.parameters(), lr=2e-5, eps=1e-8)
total_steps = len(train_dataloader) * 4
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model_bert.to(device)

for epoch in range(4):
    print(f"Epoch {epoch + 1}/{4}")
    model_bert.train()

    total_loss = 0

    for step, batch in enumerate(train_dataloader):
        batch_input_ids, batch_attention_masks, batch_labels = [item.to(device) for item in batch]

        model_bert.zero_grad()

        outputs = model_bert(input_ids=batch_input_ids, attention_mask=batch_attention_masks, labels=batch_labels)

        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model_bert.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

    avg_train_loss = total_loss / len(train_dataloader)


model_bert.eval()
predictions, true_labels = [], []

for batch in test_dataloader:
    batch_input_ids, batch_attention_masks, batch_labels = [item.to(device) for item in batch]

    with torch.no_grad():
        outputs = model_bert(input_ids=batch_input_ids, attention_mask=batch_attention_masks)

    logits = outputs.logits
    predictions.extend(torch.argmax(logits, dim=1).tolist())
    true_labels.extend(batch_labels.tolist())

import os
output_dir = '/working/bert_model/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

model_bert.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bert_model.to(device)

def get_bert_embeddings(texts, batch_size=16):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        encoded = tokenizer(
            batch,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128
        ).to(device)
        with torch.no_grad():
            outputs = bert_model(**encoded)
        batch_embeds = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings.append(batch_embeds)
    return np.vstack(embeddings)

human_texts = df[df['generated'] == 0]['clean_text'].sample(2000, random_state=42).tolist()
ai_texts = df[df['generated'] == 1]['clean_text'].sample(2000, random_state=42).tolist()

X_train = get_bert_embeddings(human_texts)

X_test = get_bert_embeddings(human_texts[:500] + ai_texts[:500])
y_test = np.concatenate([np.ones(500), -np.ones(500)])

ocsvm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)
ocsvm.fit(X_train)

y_pred = ocsvm.predict(X_test)

y_pred_binary = np.where(y_pred == 1, 1, 0)
y_true_binary = np.where(y_test == 1, 1, 0)

ocsvm.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
