import json
import unicodedata
from num2words import num2words
import dateparser
import nltk
import re
import advertools as adv
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
import simplemma
from textblob import TextBlob

# Download necessary NLTK data for preprocessing
#nltk.download("stopwords")
#nltk.download("punkt", quiet=True)

with open("Task7/config/language_mappings.json", "r") as f:
        mappings = json.load(f)

lemmer_codes = mappings["lemmer_codes"]
adv_encoding = mappings["adv_encoding"]
nltk_encoding = mappings["nltk_encoding"]

def remove_punctuation(text: str):
    """
    Function to remove punctuation.
    
    Args:
        text (str): The input text.
    
    Returns:
        str: Normalized text.
    """
    date_pattern = r'\b\d{4}-\d{2}-\d{2}\b'
    abbreviation_pattern = r'\b(?:[A-Za-z]\.[A-Za-z](?:\.[A-Za-z])*)\b'
    number_pattern = r'\b\d+(?:,\d{3})*(?:\.\d+)?\b'
    combined_pattern = f'({date_pattern})|({abbreviation_pattern})|({number_pattern})'
    text = re.sub(r'[^\w\s\-]', lambda match: '' if not re.search(combined_pattern, match.group(0)) else match.group(0), text)
    return text

def normalize_numbers(text: str):
    """
    Function to replace numbers with their word forms.
    
    Args:
        text (str): The input text.
    
    Returns:
        str: Normalized text.
    """
    def replace_number(match):
        number = match.group(0)
        try:
            return num2words(float(number))
        except ValueError:
            return number  # if num2words fails, leave the original text
    return re.sub(r'\b\d+(\.\d+)?\b', replace_number, text)

def normalize_dates(text: str):
    """
    Function to normalize dates.
    
    Args:
        text (str): The input text.
    
    Returns:
        str: Normalized text.
    """
    def replace_date(match):
        date_str = match.group(0)
        date_obj = dateparser.parse(date_str)
        if date_obj:
            return date_obj.strftime('%B %d, %Y')
        return date_str
    return re.sub(r'\b(?:\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}|\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2})\b', replace_date, text)

def remove_emoji(text: str):
    """
    Removes emojis from text.
    
    Args:
        text (str): The input text to be cleaned.
    
    Returns:
        str: Clean text.
    """
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_html(text: str):
    """
    Removes html from text.
    
    Args:
        text (str): The input text to be cleaned.
    
    Returns:
        str: Clean text.
    """
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)

def correct_spelling_and_grammar(text: str):
    """
    Corrects grammar and spelling of the given text.
    
    Args:
        text (str): The input text to be corrected.
    
    Returns:
        str: Corrected text.
    """
    blob = TextBlob(text)
    return str(blob.correct())

def remove_duplicates(text: str):
    """
    Removes duplicate words from text.
    
    Args:
        text (str): The input text.
    
    Returns:
        str: Deduplicated text.
    """

    words = text.split()
    unique_words = list(dict.fromkeys(words))  # Preserve order while removing duplicates
    return " ".join(unique_words)

def lemmatize(text:str, lang:str):
    """
    Lemmatizes the input text based on the specified language.
    
    Args:
        text (str): The input text to be lemmatized.
        lang (str): The language of the text.
    
    Returns:
        str: The lemmatized text.
    """

    if lang not in nltk_encoding:
        stop_words = adv.stopwords.get(adv_encoding.get(lang, 'english'))
    else:
        stop_words = stopwords.words(nltk_encoding.get(lang, 'english')) 
    
    words = text.split()
    
    if lang == "eng":
        lemmatizer = WordNetLemmatizer()
        wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}
        # POS tagging and lemmatization
        lemmatized_text = [
            lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) 
            for word, pos in nltk.pos_tag(words) if word not in stop_words
        ]
    else:
        lem_lang = lemmer_codes.get(lang, 'en')
        lemmatized_text = [
            simplemma.lemmatize(word, lang=lem_lang) 
            for word in words if word not in stop_words
        ]
    return ' '.join(lemmatized_text)

def preprocessing(text: str, lang: str):
    """
    Applies the full preprocessing pipeline to the input text based on the specified language.
    
    Args:
        text (str): The input text to be preprocessed.
        lang (str): The language of the text.
    
    Returns:
        str: The preprocessed text.
    """
    text = str(text).lower()    
    text = re.sub(r'[\n\t\r]+', ' ', text) # Normalize newlines and tabs to spaces
    text = remove_punctuation(text)
    text = remove_emoji(text)
    text = remove_html(text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\&\w*;', '', text)
    #text = correct_spelling_and_grammar(text)
    text = remove_duplicates(text)
    text = unicodedata.normalize('NFKD', text)
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    text = normalize_dates(text)
    #text = normalize_numbers(text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = lemmatize(text, lang)
    return text