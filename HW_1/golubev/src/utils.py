import html
import random
import re

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline

SEED = 42
N_SPLITS = 5


def load_config(path):
    if not path:
        return {}

    import yaml
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    except ImportError:
        pass


MOJIBAKE_MAP = {
    '\x89Û÷': "'", '\x89Ûª': "'",      # одинарные кавычки и апостроф
    '\x89ÛÏ': '"', '\x89Û\x9d': '"',   # двойные кавычки
    '\x89ÛÒ': ' - ', '\x89ÛÓ': ' - ',  # тире
    '\x89Û_': '...',                   # многоточие
    '\x89ã¢': ' ',                     # маркер списка
    'åÊ': ' ', 'åÈ': ' ', 'å¨': ' ',   # неразрывный пробел и прочие разделители
    'Ì©': 'e', 'Ì_': 'o',              # é (Réunion), ó (Japón)
}
MOJIBAKE_RE = re.compile('|'.join(map(re.escape, sorted(MOJIBAKE_MAP, key=len, reverse=True))))
URL_RE = re.compile(r'https?://\S*|\bhttps?\b|www\.\S+')
MENTION_RE = re.compile(r'@\w+')
HASHTAG_RE = re.compile(r'#(?=\w)')
RT_RE = re.compile(r'\bRT\b')
NON_ASCII_RE = re.compile(r'[^\x00-\x7f]')


def fix_html(t):
    prev = None
    while prev != t:
        prev, t = t, html.unescape(t)

    return t


def clean_text(t, url_token='URL', user_token='@USER'):
    t = fix_html(t)
    t = MOJIBAKE_RE.sub(lambda m: MOJIBAKE_MAP[m.group()], t)
    t = URL_RE.sub(f' {url_token} ', t)
    t = MENTION_RE.sub(user_token, t)
    t = NON_ASCII_RE.sub(' ', t)
    t = t.replace('#', ' ')
    t = RT_RE.sub(' ', t)
    
    return ' '.join(t.split())


def normalize(t):
    t = html.unescape(t).lower()
    t = re.sub(r'http\S+|@\w+|#', ' ', t)
    t = re.sub(r'[^a-z ]', ' ', t)
    
    return ' '.join(t.split())


def preprocess(df):
    df = df.copy()
    df['has_url'] = df['text'].str.contains(URL_RE).astype(int)
    df['has_mention'] = df['text'].str.contains(MENTION_RE).astype(int)
    df['has_hashtag'] = df['text'].str.contains(HASHTAG_RE).astype(int)
    df['text_clean'] = df['text'].map(clean_text)
    df['text_bertweet'] = df['text'].map(lambda t: clean_text(t, url_token='HTTPURL'))
    df['text_norm'] = df['text'].map(normalize)
    df['keyword'] = df['keyword'].str.replace('%20', ' ', regex=False).fillna('unknown')
    
    return df


def load_data(path):
    return preprocess(pd.read_csv(path))


def make_folds(df, n_splits=N_SPLITS, seed=SEED):
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    groups = df['text_norm'].factorize()[0]

    return list(sgkf.split(df, df['target'], groups))


def best_threshold(y_true, proba, grid=np.arange(0.05, 0.96, 0.01)):
    scores = [f1_score(y_true, (proba >= t).astype(int), zero_division=0) for t in grid]
    
    return float(grid[int(np.argmax(scores))])


def tune_threshold_inner(model_factory, df_tr, n_inner=3, seed=SEED):
    oof = np.zeros(len(df_tr))
    for in_tr, in_va in make_folds(df_tr, n_splits=n_inner, seed=seed):
        m = model_factory()
        m.fit(df_tr.iloc[in_tr])
        oof[in_va] = m.predict_proba(df_tr.iloc[in_va])
    
    return best_threshold(df_tr['target'].values, oof)


def fit_with_threshold(model_factory, df, tune_threshold=True):
    model = model_factory()
    model.fit(df)
    if tune_threshold:
        thr = tune_threshold_inner(model_factory, df)
    
    else:
        thr = getattr(model, 'threshold', 0.5)
    
    return model, thr


def classification_metrics(y, pred):
    return {'f1': f1_score(y, pred, zero_division=0),
            'precision': precision_score(y, pred, zero_division=0),
            'recall': recall_score(y, pred, zero_division=0)}


def cv_evaluate(model_factory, df, name, folds, tune_threshold=True, verbose=True):
    rows = []
    oof = np.zeros(len(df))
    for k, (tr_idx, va_idx) in enumerate(folds):
        df_tr, df_va = df.iloc[tr_idx], df.iloc[va_idx]
        model, thr = fit_with_threshold(model_factory, df_tr, tune_threshold)
        proba = model.predict_proba(df_va)
        oof[va_idx] = proba
        pred = (proba >= thr).astype(int)
        row = {'model': name, 'fold': k, 'threshold': round(thr, 2),
               **classification_metrics(df_va['target'].values, pred)}
        rows.append(row)
    
        if verbose:
            print(f'[{name}] fold {k}: f1={row["f1"]:.3f} '
                  f'(precision={row["precision"]:.3f}, recall={row["recall"]:.3f}, thr={thr:.2f})')
    
    return pd.DataFrame(rows), oof


def eval_oof(proba, df, folds, name):
    y = df['target'].values
    rows = []
    for k, (_, va_idx) in enumerate(folds):
        other = np.concatenate([va for j, (_, va) in enumerate(folds) if j != k])
        thr = best_threshold(y[other], proba[other])
        pred = (proba[va_idx] >= thr).astype(int)
        rows.append({'model': name, 'fold': k, 'threshold': round(thr, 2),
                     **classification_metrics(y[va_idx], pred)})
    
    return pd.DataFrame(rows)


def compare_models(res_a, res_b):
    a = res_a.sort_values('fold')['f1'].values
    b = res_b.sort_values('fold')['f1'].values
    diff = b - a
    spread = max(a.std(ddof=1), b.std(ddof=1))
    
    return {'mean_a': a.mean(), 'mean_b': b.mean(), 'mean_diff': diff.mean(),
            'spread': spread, 'wins_all_folds': bool((diff > 0).all()),
            'confirmed': bool((diff > 0).all() and diff.mean() > spread)}


class ConstantBaseline:
    def fit(self, df):
        return self

    def predict_proba(self, df):
        return np.ones(len(df))


class RandomBaseline:
    def __init__(self, seed=SEED):
        self.rng = np.random.default_rng(seed)

    def fit(self, df):
        self.prior = df['target'].mean()
        return self

    def predict_proba(self, df):
        return (self.rng.random(len(df)) < self.prior).astype(float)


class KeywordBaseline:
    def __init__(self, smoothing=10):
        self.m = smoothing

    def fit(self, df):
        self.prior = df['target'].mean()
        stats = df.groupby('keyword')['target'].agg(['sum', 'count'])
        self.enc = (stats['sum'] + self.m * self.prior) / (stats['count'] + self.m)

        return self

    def predict_proba(self, df):
        return df['keyword'].map(self.enc).fillna(self.prior).values


class TfidfLogReg:
    def __init__(self, ngram_range=(1, 2), min_df=2, C=1.0):
        self.pipe = make_pipeline(
            TfidfVectorizer(lowercase=True, ngram_range=ngram_range, min_df=min_df,
                            sublinear_tf=True, strip_accents='unicode'),
            LogisticRegression(C=C, max_iter=1000, random_state=SEED),
        )

    def fit(self, df):
        self.pipe.fit(df['text_clean'], df['target'])
        return self

    def predict_proba(self, df):
        return self.pipe.predict_proba(df['text_clean'])[:, 1]


class TfidfWordCharLogReg:
    def __init__(self, C=1.0, use_keyword=True):
        self.C = C
        self.use_keyword = use_keyword

    def _word_text(self, df):
        if not self.use_keyword:
            return df['text_clean']
    
        return 'kw_' + df['keyword'].str.replace(' ', '_') + ' ' + df['text_clean']

    def _transform(self, df):
        return hstack([self.word_vec.transform(self._word_text(df)),
                       self.char_vec.transform(df['text_clean'])]).tocsr()

    def fit(self, df):
        self.word_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                        strip_accents='unicode').fit(self._word_text(df))
        self.char_vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=3,
                                        sublinear_tf=True).fit(df['text_clean'])
        self.clf = LogisticRegression(C=self.C, max_iter=2000, random_state=SEED)
        self.clf.fit(self._transform(df), df['target'])
        return self

    def predict_proba(self, df):
        return self.clf.predict_proba(self._transform(df))[:, 1]


class TransformerClassifier:

    def __init__(self, model_name='vinai/bertweet-base', text_col='text_bertweet',
                 max_len=64, batch_size=32, lr=2e-5, epochs=3, warmup=0.1,
                 weight_decay=0.01, holdout=0.1, seed=SEED, verbose=True):
        self.model_name, self.text_col = model_name, text_col
        self.max_len, self.batch_size, self.lr, self.epochs = max_len, batch_size, lr, epochs
        self.warmup, self.weight_decay, self.holdout = warmup, weight_decay, holdout
        self.seed, self.verbose = seed, verbose

    def _batches(self, texts, labels=None, shuffle=False):
        import torch
        from torch.utils.data import DataLoader

        def collate(b):
            enc = self.tok([texts[i] for i in b], truncation=True, max_length=self.max_len,
                           padding=True, return_tensors='pt')
            if labels is not None:
                enc['labels'] = torch.tensor([labels[i] for i in b])

            return enc

        g = torch.Generator().manual_seed(self.seed)
        return DataLoader(np.arange(len(texts)), batch_size=self.batch_size, shuffle=shuffle,
                          collate_fn=collate, generator=g)

    def _proba(self, texts):
        import torch
        self.model.eval()
        out = []
        with torch.no_grad():
            for batch in self._batches(texts):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                logits = self.model(**batch).logits
                out.append(torch.softmax(logits, dim=-1)[:, 1].float().cpu().numpy())
        
        return np.concatenate(out)

    def fit(self, df):
        import torch
        from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                                  get_linear_schedule_with_warmup)

        set_seed(self.seed)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tok = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=2).to(self.device)

        gss = GroupShuffleSplit(n_splits=1, test_size=self.holdout, random_state=self.seed)
        tr_idx, ho_idx = next(gss.split(df, groups=df['text_norm']))
        tr, ho = df.iloc[tr_idx], df.iloc[ho_idx]
        tr_texts, tr_y = tr[self.text_col].tolist(), tr['target'].tolist()
        ho_texts, ho_y = ho[self.text_col].tolist(), ho['target'].values

        loader = self._batches(tr_texts, tr_y, shuffle=True)
        steps = len(loader) * self.epochs
        no_decay = ['bias', 'LayerNorm.weight']
        params = [
            {'params': [p for n, p in self.model.named_parameters()
                        if not any(nd in n for nd in no_decay)],
             'weight_decay': self.weight_decay},
            {'params': [p for n, p in self.model.named_parameters()
                        if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
        ]
        opt = torch.optim.AdamW(params, lr=self.lr)
        sched = get_linear_schedule_with_warmup(opt, int(self.warmup * steps), steps)
        use_amp = self.device == 'cuda'
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

        best_f1, best_state = -1, None
        for epoch in range(self.epochs):
            self.model.train()
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                    loss = self.model(**batch).loss
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()

            p = self._proba(ho_texts)
            thr = best_threshold(ho_y, p)
            f1 = f1_score(ho_y, (p >= thr).astype(int))
            if self.verbose:
                print(f'  epoch {epoch + 1}: holdout f1={f1:.3f} (threshold={thr:.2f})')
            if f1 > best_f1:
                best_f1, self.threshold = f1, thr
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

        self.model.load_state_dict(best_state)
        return self

    def predict_proba(self, df):
        return self._proba(df[self.text_col].tolist())

    def save(self, path):
        """Сохраняет веса и токенизатор в формате Hugging Face."""
        self.model.save_pretrained(path)
        self.tok.save_pretrained(path)


MODELS = {
    'constant': (ConstantBaseline, False),
    'random': (RandomBaseline, False),
    'keyword': (KeywordBaseline, True),
    'tfidf': (TfidfLogReg, True),
    'tfidf_wordchar': (TfidfWordCharLogReg, True),
    'bertweet': (TransformerClassifier, False),
}
