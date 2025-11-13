import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.probability import FreqDist
from nltk.lm import Laplace
from nltk.lm.preprocessing import padded_everygram_pipeline
import spacy
import re
import math
from collections import Counter

# --- Language Detection ---
def is_arabic(text):
    arabic_char_count = len(re.findall(r'[\u0600-\u06FF]', text))
    total_char_count = len(text)
    return arabic_char_count / max(total_char_count, 1) > 0.2

# --- Preprocess & Tokenize ---
def preprocess_and_tokenize(text, lang):
    if lang == 'en':
        text = text.lower()
        tokens = word_tokenize(text)
        tokens = [w for w in tokens if w.isalnum()]
    elif lang == 'ar':
        # Normalize and simple tokenization
        text = re.sub(r'[^\u0600-\u06FF\s]', '', text)
        tokens = re.findall(r'[\u0600-\u06FF]+', text)
    else:
        tokens = []
    return tokens

# --- Rule-based POS Tagging (for Arabic) ---
def arabic_pos_rulebased(tokens):
    pos_tags = []
    for token in tokens:
        if re.match(r'^[ال].+', token):       # starts with "ال" (the) → likely noun
            pos_tags.append((token, 'NOUN'))
        elif len(token) <= 2:                 # short words → possible prepositions or particles
            pos_tags.append((token, 'PART'))
        elif token.endswith('ة') or token.endswith('ات'):
            pos_tags.append((token, 'NOUN'))
        elif token.endswith('ي') or token.endswith('ك') or token.endswith('هم'):
            pos_tags.append((token, 'PRON'))
        elif token.endswith('ون') or token.endswith('ين'):
            pos_tags.append((token, 'VERB'))
        else:
            pos_tags.append((token, 'OTHER'))
    return pos_tags

def pos_tagging(tokens, lang):
    if lang == 'en':
        return nltk.pos_tag(tokens)
    elif lang == 'ar':
        return arabic_pos_rulebased(tokens)
    return []

# --- N-gram Analysis ---
def n_gram_analysis(tokens, n=2):
    n_grams = list(ngrams(tokens, n))
    freq_dist = FreqDist(n_grams)
    return freq_dist.most_common(10)

# --- Perplexity (simplified) ---
def calculate_perplexity(tokens):
    if len(tokens) < 5:
        return "Not enough tokens to calculate perplexity."
    train_data, vocab = padded_everygram_pipeline(2, [tokens])
    lm = Laplace(2)
    lm.fit(train_data, vocab)
    test_data, _ = padded_everygram_pipeline(2, [tokens])
    ppx_list = []
    for sent_ngrams in test_data:
        try:
            val = lm.perplexity(sent_ngrams)
            if not math.isinf(val):
                ppx_list.append(val)
        except ZeroDivisionError:
            continue
    return f"{sum(ppx_list)/len(ppx_list):.2f}" if ppx_list else "N/A"

# --- spaCy for English ---
@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.error("spaCy model not found.")
        return None
nlp = load_spacy_model()

# --- Rule-based Arabic NER ---
def arabic_ner_rulebased(text):
    entities = []
    # Example simple rules
    if 'الإمارات' in text or 'دبي' in text or 'أبوظبي' in text:
        entities.append(('الإمارات', 'LOC'))
    if 'محمد' in text or 'خالد' in text:
        entities.append(('محمد', 'PER'))
    if 'جامعة' in text:
        entities.append(('جامعة', 'ORG'))
    return entities if entities else [('—', '—')]

def ner_analysis(text, lang):
    if lang == 'en' and nlp:
        doc = nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]
    elif lang == 'ar':
        return arabic_ner_rulebased(text)
    return []

# --- Streamlit UI ---
def main():
    st.set_page_config(page_title="News NLP Web Demo", page_icon="🗞️", layout="wide")
    
    # Hero header
    st.markdown("# 🗞️ News NLP Web Demo — Arabic–English pipeline (paste text or URL)")
    
    # Create tabs
    tab_demo, tab_about, tab_credits = st.tabs(["Demo", "About", "Credits"])
    
    with tab_demo:
        st.title("📰 Arabic–English News NLP Pipeline")
        st.markdown("Enter newspaper text below to analyze using rule-based Arabic + spaCy/NLTK English pipeline.")
        
        text = st.text_area("Enter text:", height=250, value="شركة أبل هي شركة تكنولوجيا مقرها كاليفورنيا. محمد يعمل هناك.")
        if st.button("Apply NLP Pipeline"):
            lang = 'ar' if is_arabic(text) else 'en'
            st.info(f"Detected language: **{'Arabic' if lang == 'ar' else 'English'}**")
            
            tokens = preprocess_and_tokenize(text, lang)
            if not tokens:
                st.warning("No tokens found.")
                return
            
            # POS
            with st.expander("1️⃣ POS Tagging", expanded=True):
                pos_tags = pos_tagging(tokens, lang)
                st.dataframe([{"Token": t, "POS": p} for t, p in pos_tags])
            
            # N-gram
            with st.expander("2️⃣ Top 10 Bigrams", expanded=True):
                ngrams_list = n_gram_analysis(tokens, n=2)
                st.dataframe([{"Bigram": " ".join(bg), "Freq": f} for bg, f in ngrams_list])
            
            # Perplexity
            with st.expander("3️⃣ Perplexity (Demo)", expanded=True):
                score = calculate_perplexity(tokens)
                st.write(f"**Perplexity:** {score}")
            
            # NER
            with st.expander("4️⃣ Named Entity Recognition (NER)", expanded=True):
                ents = ner_analysis(text, lang)
                st.dataframe([{"Entity": e, "Type": t} for e, t in ents])
    
    with tab_about:
        st.markdown("""
        This demo runs an Arabic–English News NLP pipeline. Paste text or a URL to detect language, classify/score, and view key terms and a short summary. Student prototype—results may be imperfect.
        """)
    
    with tab_credits:
        st.markdown("""
        **Group Members:** Alia Al Ali; Aya Ehab; Rana Kamal Eldin; Reem Bin Haider; Salma Amarah
        
        **Affiliation:** The British University in Dubai (BUiD)
        
        **Copyright:** © 2025 Alia Al Ali, Aya Ehab, Rana Kamal Eldin, Reem Bin Haider, Salma Amarah. All rights reserved.
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("**Built with Streamlit • Demo v1.0**")

if __name__ == "__main__":
    main()
