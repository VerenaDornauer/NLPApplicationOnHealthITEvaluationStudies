# Application of Natural Language Processing on Health IT evaluation studies

Basic informations and code for the application of Natural Language Processing on Health IT evaluation studies with Python

Methodological steps:
24 Health IT evaluation studies were selected (see txt file). The aim was to retrieve: evaluated application system, features of the evaluated application system, organizational unit and users of the evaluated application system, software product name, author of the study, the used study method and the outcome criteria of the study. The Gold Standard was manually retrieved these key elements. We compared the effectiveness of the methods by comparing the extracted key elements with outcome from manual data extraction. The efficiency was measured by observing the coding task due to complexity of the code and the reusability of the Python code. 

The corpora of the 24 health IT studies were created with the help of word and finally created imported to the IDE PyCharm. Afterwards the four natural language processing methods (Bag-of-Words, Term-Frequency-Inverse-Document-Frequency, Latent Dirichlet Allocation Topic Modelling and Named Entity Recognition) were applied to the 24 health IT evaluation studies (here abstract section, introduction and methods section).

The code for importing word documents in PyCharm:

```
from docx import Document

doc = open("*study_path*", "rb")

document = Document(doc)
study = ""
for para in document.paragraphs:
    study += para.text
```

The code for Bag-of-Words:

```
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from collections import Counter
import re

pattern1 = "\["
pattern2 = "\]"
study_clean_mid = re.sub(pattern1, " ", study)
study_clean = re.sub(pattern2, " ", study_clean_mid)
tokenizer = RegexpTokenizer(r'[a-zA-z]+')
study_word_tokenize = tokenizer.tokenize(study_clean)

english_stops = set(stopwords.words('english'))
study_without_stops = [word for word in study_word_tokenize if word not in english_stops]

lower_study_preprocessed = [t.lower() for t in study_without_stops]

bag_of_words_study = Counter(lower_study_preprocessed)

print(bag_of_words_study.most_common(15))
```

The code for Term-Frequency-Inverse-Document-Frequency:

```
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

study_corpus = sent_tokenize(study)

tfidf_vectorizer=TfidfVectorizer(stop_words='english',
                                 max_features=15)

X = tfidf_vectorizer.fit_transform(study_corpus)
print(tfidf_vectorizer.get_feature_names())
```

The code for Latent Dirichlet Allocation Topic Modeling:

```
import re
from nltk.tokenize import RegexpTokenizer, word_tokenize
from nltk.corpus import stopwords
import gensim
import pyLDAvis.gensim

pattern1 = "\["
pattern2 = "\]"
study_clean_mid = re.sub(pattern1, " ", study)
study_clean = re.sub(pattern, " ", study_clean_mid)
tokenizer = RegexpTokenizer(r'[a-zA-z]+')
study_word_tokenize = tokenizer.tokenize(study_clean)

english_stops = set(stopwords.words('english'))
study_without_stops = [word for word in study_word_tokenize if word not in english_stops]

dictionary = gensim.corpora.Dictionary([study_without_stops])
dictionary.filter_extremes(no_below=1, no_above=20)
corpus = [dictionary.doc2bow(token) for token in [study_without_stops]]
lda_model = gensim.models.LdaModel(corpus=corpus, num_topics=2,
id2word=dictionary, chunksize=20)

print(lda_model)
print(lda_model.print_topics())

vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary=lda_model.id2word)
pyLDAvis.show(vis)
```

The code for Named Entity Recognition:

```
import re
import spacy

pattern = '\d+ | \% | \$ | \d+\.\d+'

study_clean = re.sub(pattern, "", study)

nlp = spacy.load('en_core_web_sm')

doc = nlp(study_clean)

ent_list=['PERSON', 'ORG', 'PRODUCT']

for ent in doc.ents:
        if ent.label_ in ent_list:
            print(ent.label_, ent.text)
```

The results are meanwhile submitted to the ICIMTH 2020. Please contact me for further details about used Health IT evaluation studies or .py files.
