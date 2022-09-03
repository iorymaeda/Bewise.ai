import re

import torch
import aiohttp
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased-conversational")
model = AutoModel.from_pretrained("DeepPavlov/rubert-base-cased-conversational")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)


def avg_pooling(window: torch.Tensor, seq_len: torch.Tensor, key_padding_mask: torch.Tensor)  -> torch.FloatTensor:
    """Помогает пулить padded последовательности
    
    source: https://github.com/iorymaeda/ArcDOTA/blob/master/utils/nn/prematch.py 
    
    :param window: - window/array to pool
    | window | : (batch_size, seq_len, d_model)
    
    :param seq_len: - array with seq_len per sample in batch
    | seq_len | : (batch_size)
    
    :param key_padding_mask: - array with padded mask
    | key_padding_mask | : (batch_size, seq_len)
    
    """
    
    # multiply window by mask - zeros all padded tokens
    pooled = torch.mul(window, key_padding_mask.unsqueeze(2))
    # |pooled| : (batch_size, seq_len, d_model)

    # sum all elements by seq_len dim
    pooled = pooled.sum(dim=1)
    # |pooled| : (batch_size, d_model)

    # divide samples by by its seq_len, so we will get mean values by each sample
    pooled = pooled / seq_len.unsqueeze(1)
    # |pooled| : (batch_size, d_model)

    return pooled


def split(text: str) -> str:
    return text.replace("-", " ").replace("_", " ").split()

def crop_text_to_patches(text: str, slice_range:int=2) -> list:
    query_slices = []
    query_words = split(text)
    
    if len(query_words) < slice_range:
        query_slices.append(text)
        
    else:
        for batch in range(0, len(query_words)-(slice_range-1)):
            s = ""
            for _ in range(slice_range):
                s+= query_words[batch + _] + " "
            s = s.strip()

            query_slices.append( s )
            
    return query_slices

def dict_to_device(d, device):
    return {k: v.to(device) for k, v in d.items()}

@torch.no_grad()
def classify(query: str, anchor: list, slice_range=2) -> float:
    query = query.lower()
    query_slices = crop_text_to_patches(query, slice_range=slice_range)
    
    text = anchor + query_slices
    tokens = tokenizer(text, return_tensors='pt', padding=True)
    out = model(**dict_to_device(tokens, device))
    out = dict_to_device(out, 'cpu')
    
    # embs = out['last_hidden_state'].mean(dim=1)
    embs = avg_pooling(out['last_hidden_state'], tokens['attention_mask'].sum(1), tokens['attention_mask'])
    normed = embs / torch.norm(embs, dim=1, keepdim=True)
    
    _anchor, _query = normed[:len(anchor)], normed[len(anchor):]
    max_ = (_anchor @ _query.T).max()
    
    if slice_range == 1:
        return max_
    else:
        return max(max_, classify(query, anchor, slice_range-1))


def to_camel_case(text):
    if len(text) == 0: return text

    s = split(text)
    output = ""
    for _t in s:
        output += _t.capitalize()
        output += " "
    return output.rstrip()


def preproc(replic: str) -> dict[str, list[int]]:
    # Кропаем и подаём патчами т.к. модель глупенькая и на больших предложениях путаеться 
    return crop_text_to_patches(to_camel_case(replic), slice_range=5) 

async def get_ner(replic: str): 
    async with aiohttp.ClientSession() as session:
        async with session.post(url="http://dp-ner:5000/model", json={ "x": preproc(replic)}) as response:
            ner = await response.json()
        
    # Куча постпроцессинга
    ners = [[] for _ in range(len(split(replic)))]
    words = []
    for idx, v in enumerate(ner):
        t, n = v

        if idx == 0: words += t
        else: words += t[-1:]

        n = [re.sub("\D-", "", ner_v) for ner_v in n]
        for jdx, ner_v in enumerate(n):
            ners[idx+jdx].append(ner_v)

    align = []
    for i in ners:
        d = {}
        for entity in i:
            if entity in d:
                d[entity] += 1
            else:
                d[entity] = 1

        for key in d:
            d[key] /= len(i)
        align.append(d)
        
    NER = []
    for t in align:
        entity = max(t, key=t.get)
        if t[entity] >= 0.5:
            NER.append(entity)
        else:
            NER.append('O')

    collected = {}
    _previous_num = 0
    _previous_tag = NER[0]
    for idx, tag in enumerate(NER):
        if idx > 0:
            if tag != _previous_tag:
                if _previous_tag not in collected:
                    collected[_previous_tag] = []

                collected[_previous_tag].append([_previous_num, idx])

                _previous_num = idx
                _previous_tag = tag

    if _previous_tag not in collected:
        collected[_previous_tag] = []

    collected[_previous_tag].append([_previous_num, len(NER)])
    
    if 'O' in collected:
        del collected['O']
    
    return collected

async def extract_ner(replic: str):
    ners = await get_ner(replic)
    splitted_replic = split(replic)
    for key in ners:
        for idx, pos in enumerate(ners[key]):
            s = ""
            _replic = splitted_replic[pos[0]:pos[1]]
            for word in _replic:
                s+= word
                s+= " "
            s = s.strip()

            ners[key][idx] = s
    return ners

async def process_df(df: pd.DataFrame) -> dict:
    for idx, replic in enumerate(tqdm(df['text'])): 
        replic: str
        replic = replic.strip()
        
        # --------------------------------------------------------------------- #
        ners = await extract_ner(replic)
        greetings = classify(replic, slice_range=2, anchor=['приветствие', 'здравствуйте', 'добрый день', 'доброе утро', 'добрый вечер'])
        farewells = classify(replic, slice_range=2, anchor=['до свидания', 'всего хорошего', 'прощай', 'прощайте', 'хорошего дня', 'хорошего вечера'])
        
        greetings = greetings.item()
        farewells = farewells.item()
        
        # --------------------------------------------------------------------- #
        # speech2text съездает некоторые слова, отдельно обработаем такие случаи
        words = split(replic.lower())[0]
        if words[0] in ['алло', 'ало']: del words[0]
        
        if words[0] in ['добрый', 'доброе']:
            greetings = 0.9 if greetings < 0.9 else greetings
        
        # --------------------------------------------------------------------- #
        df.loc[idx, 'greetings_score'] = greetings
        df.loc[idx, 'farewells_score'] = farewells

        for entity in ners:
            df.loc[idx, entity] = ners[entity][0]

    # --------------------------------------------------------------------- #
    # Модель очень тригерится на Алло, думает это имена
    # К сожалению скоры deeppavlov не выдаёт
    df.loc[df['PER'] == 'Алло', 'PER'] = float('nan')
    df["PER"] = df["PER"].apply(lambda x:  x.lower().replace("алло", "") if type(x) is str else x)

    # Это нам не надо
    df.drop(['LOC'], axis=1, inplace=True)

    # --------------------------------------------------------------------- #
    # Что-то мне кажется что роль менеджера и клиента перепутаны
    # Буду считать всё для клиентов представляю что это менеджеры, к тому же этот нюанс можно уточнить и легко исправить

    # Ещё один пост процессинг, обычно здороваются в начале диалога, предпологаю что компанию тоже стоит назвать в начале диалога,
    # а прощаются в конце, таким образом отчистим от мусора который выдал deeppavlov
    role = 'client'
    greetings_index = []
    farewells_index = []

    names_index = []
    organization_index = []

    for dlg_id  in df['dlg_id'].unique():
        corpus = df[(df['dlg_id'] == dlg_id) & (df['role'] == role)]
        
        # Приветсвие обычно в начале
        slice_ = corpus.iloc[:2]['greetings_score']
        for index in (slice_[slice_ >= 0.85]).index:
            greetings_index.append(index)
            break
            
        # Прощание обыччно в конце
        slice_ = corpus.iloc[-4:]['farewells_score']
        for index in (slice_[slice_ >= 0.85]).index:
            farewells_index.append(index)
            break
        
        # Если менеджер не назвал своего имени в начале - он не вежда
        slice_ = corpus.iloc[:4]['PER']
        for index in slice_.index:
            names_index.append(index)
            
        slice_ = corpus.iloc[:4]['ORG']
        for index in slice_.index:
            organization_index.append(index)
            
    df['greetings'] = False
    df['farewells'] = False
    df.loc[greetings_index, 'greetings'] = True
    df.loc[farewells_index, 'farewells'] = True

    df['PER_name'] = float('nan')
    df['ORG_name'] = float('nan')
    df.loc[names_index, 'PER_name'] = df.loc[names_index, 'PER']
    df.loc[organization_index, 'ORG_name'] = df.loc[organization_index, 'ORG']

    # --------------------------------------------------------------------- #
    # Собираем в словарь всё остальное
    dlf_output = {}
    for dlg_id  in df['dlg_id'].unique():
        df_slice = df[(df['dlg_id'] == dlg_id)]
        
        greeting = df_slice['greetings'].any()
        farewell = df_slice['farewells'].any()
        
        PER_name = df_slice[df_slice['role'] == role]['PER_name'].dropna().unique().tolist()
        ORG_name = df_slice[df_slice['role'] == role]['ORG_name'].dropna().unique().tolist()
        
        greeting_text = df_slice[df_slice['greetings']]['text']
        farewell_text = df_slice[df_slice['farewells']]['text']
        
        greeting_text = greeting_text.values[0] if len(greeting_text) else ''
        farewell_text = farewell_text.values[0] if len(farewell_text) else ''
        
        dlf_output[dlg_id] = {
            'greeting': greeting,
            'farewell': farewell,
            'farewell_text': farewell_text,
            'greeting_text': greeting_text,
            'PER_name': PER_name,
            'ORG_name': ORG_name,
            'is_polite': (greeting and farewell)
        }

    return dlf_output