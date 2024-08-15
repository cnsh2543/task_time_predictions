import re
from sklearn.preprocessing import LabelEncoder
import nltk
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
import spacy
# from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
import json
import os 
from datetime import datetime


nlp = spacy.load('en_core_web_md')

# Set the NLTK data directory
nltk.data.path.append(os.getenv('NLTK_DATA', '/nltk_data'))

# # Load spaCy model from the specified directory
# nlp = spacy.load(os.getenv('SPACY_DATA') + '/en_core_web_md')


# model_path = "./model"
# model = SentenceTransformer(model_path, cache_folder='/tmp/sentence_transformers_cache')

def lambda_handler(event, context):
    '''Provide an event that contains the following keys:

      - message: contains the text 
    '''
    try:
        inputText = event['questionInfo']
        lowerBound, upperBound = prediction(inputText)

        return {
            'statusCode': 200,
            'body': json.dumps({"upperBound" : upperBound,
                                "lowerBound": lowerBound})
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": repr(e)})
        } 



## to process y
# clean up the formatting of questions, and break them into sentences
def clean_text(text):

    modified_text = re.sub(r'^o |\\n|\n|&#x20;|\s+|(?<=i\.e)\.|(?<=e\.g)\.', ' ', text)
    modified_text = re.sub(r'\$\$', r'$', modified_text, flags=re.DOTALL)
    modified_text = re.sub(r'!?\[[^\]]*\]\([^\)]+\)', ' image ', modified_text)
    modified_text = re.sub(r'\|.*\||\<.*?\>|```[\s\S]*?```', '[embedded]', modified_text)
    return modified_text

def clean_latex(text):
   
    text = clean_text(text)    
    return text


def clean_and_restore_latex(text):
    text = clean_text(text)
    latex_segments = re.findall(r'\$ .*? *\$|\$.*?\$', text, flags=re.DOTALL)
    for i, pattern in enumerate(latex_segments):
        text = text.replace(pattern, f' latex{i} ')
    text = re.sub(r'[\]\[*()\\]', '', text)
    for i, pattern in enumerate(latex_segments):
        pattern = pattern.replace('.',',')
        text = text.replace(f'latex{i}', pattern)
    return text

# to distill questions from the entire problem statement
# List of target verbs to check within sentences
target_verbs = {
    "add", "adjust", "advise", "analyse", "apply", "approximate", "arrange", 
    "build", "calculate", "change", "check", "choose", "combine", "comment", "complete",
    "compute", "consider", "construct", "convert", "count", "create", "deduce", "define",
    "demonstrate", "derive", "describe", "determine", "develop", "differentiate", "dimensionalise",
    "distinguish", "divide", "draw","discuss", "eliminate", "enter", "estimate", "evaluate", "expand",
    "explain", "explore", "express", "extract", "fill", "find", "follow", "formulate",
    "generate", "give", "how", "identify", "illustrate", "implement", "improve", "include",
    "infer", "insert", "integrate", "introduce", "investigate", "justify", "label", "make",
    "mark", "maximize", "measure", "minimize", "modify", "multiply", "mutate", "normalise",
    "obtain", "perform", "plot", "predict", "propose", "provide",
    "rearrange", "recalculate", "reduce", "remove", "replace", "replicate", "reproduce",
    "retain", "rewrite", "select", "separate", "share", "show", "simplify", "sketch",
    "solve", "specify", "state", "submit", "subtract", "sum", "tabulate", "transform",
    "translate", "try", "update", "use", "verify", "what", "when", "which", "who", "why", "write"
}


# Function to detect if a sentence is a question based on given criteria
def is_question(sentence):
    doc = nlp(sentence)
    # Check if the sentence ends with a question mark
    if sentence.strip().endswith("?"):
        return True
    # Check if the sentence starts with a verb
    if doc[0].pos_ == "VERB":
        return True
    
    question_words = ["who", "what", "when", "where", "why", "how"]
    first_word = doc[0].text.lower()
    if first_word in question_words:
        return True
    
    for token in doc:
        if token.lemma_.lower() in target_verbs and token.pos_ == "VERB" and token.dep_ == "ROOT" and (token.tag_ == "VB" or token.tag_ == 'VBP'):
            return True
        
    return False

def lenient_is_question(sentence):
    doc = nlp(sentence)
    for token in doc:
        if token.lemma_.lower() in target_verbs:
            return True
    return False

def extract_term_title(text, terms):
    doc = nlp(text)
    tokens = []

    for term in terms:
        term_tokens = term.split()
        term_length = len(term_tokens)
        for i in range(len(doc) - term_length + 1):
            if all(doc[i+j].lemma_.lower() == term_tokens[j] for j in range(term_length)):
                tokens.append(term)
                break

    return ' '.join(tokens)

def extract_term_solution(text, terms):
    doc = nlp(text)
    tokens = []
    tokens = [j.lemma_.lower()  for j in doc if j.lemma_.lower() in terms]
    return ' '.join(tokens)

def extract_question_verbs(text):
    fivew1h_tags = {'WP', 'WP$', 'WRB', 'WDT'}
    doc = nlp(text)
    # tokens = [token.lemma_ for token in doc if token.text.lower() in ['fourier','series']]
    tokens = [token.lemma_ for token in doc if token.pos_ in ('VERB') and token.dep_ == "ROOT" and token.tag_ in ("VBP","VB")] + [token.text for token in doc if token.tag_ in fivew1h_tags]
    # tokens =  [token.text for token in doc if token.tag_ in fivew1h_tags]

    return ' '.join(tokens)

# to clean and tokenize the text
def custom_tokenize(text):
    
    modified_text = re.sub(r'\\\$', '', text)
    modified_text = re.sub(r'\n', ' ', text)
    latex_segments = re.findall(r'\$\$.*?\$\$|\$.*?\$', modified_text, re.DOTALL)
    modified_text = re.sub(r'!?\[[^\]]*\]\([^\)]+\)|\|.*\||\<.*?\>|```[\s\S]*?```', '[embedded]', modified_text)
    phrases = re.split(r'(\$\$[\s\S]*?\$\$|\$.*?\$)', modified_text)
    latex_tokens, tokens = [], []
    for phrase in phrases:
        if phrase in latex_segments:
            tokens.append("[latex]")  
            latex_tokens.append(phrase)
        else:
            # tokens.extend(nltk.word_tokenize(phrase))  
            tokens.extend(phrase.split(' '))  
    return tokens, latex_tokens

# count the number of various operations
def element_counter(text):
    operator = r'\\approx|\\parallel|\\nonumber|\\cdot|[<>+-=~]|\\times|\\div|\\pm|\\sum|\\sqrt|\\root||\\equiv|\\ne|\\neq|\\leq|\\le|\\limit|\\in|\\implies|\\gg|\\geq|\\ge|\\ll'

    num_operator = len(re.findall(operator,text))
    text = re.sub(r'\s+','',text)
    return num_operator, text

def transform(self, data):
    if not self.fitted:
        raise ValueError("The label encoder has not been fitted yet.")
    
    unique_labels = set(self.label_encoder.classes_)
    transformed_data = []
    
    for item in data:
        if item in unique_labels:
            transformed_data.append(self.label_encoder.transform([item])[0])
        else:
            transformed_data.append(-1)  # Default value for unseen labels
    
    return np.array(transformed_data)

# to use for test set to encode
def prediction(df):

    # df["module"] = re.match(r'^[A-Z]{3,}', df["modulename"].split(' ')[0]).group() if re.match(r'^[A-Z]{3,}', df["modulename"].split(' ')[0]) else 'OTHERS'
    df["level"] =  re.match(r'^[A-Z]+(\d)', df["modulename"].split(' ')[0]).group(1) if re.match(r'^[A-Z]+(\d)', df["modulename"].split(' ')[0]) else '4'

    # with open('le_module.pkl', 'rb') as file:
    #     le_module = pickle.load(file)
    # if df["module"] not in le_module.classes_:
    #     df["module"] = 0
    # else:
    #     df["module"] = le_module.transform([df["module"]])[0]

    print("Start", datetime.now().strftime("%H:%M:%S"))
    with open('le_level.pkl', 'rb') as file:
        le_level = pickle.load(file)
    if df["level"] not in le_level.classes_:
        df["level"] = '4'
    else:
        df["level"] = le_level.transform([df["level"]])[0]

    # concat all the question text
    df["total_text"] = df["masterContent"] + ' ' + df["partContent"]

    # impute partContent with masterContent if completely empty
    # pattern = r"^(\n)*$"
    # df['partContent'] = df['masterContent'] if re.match(pattern, str(df['partContent'])) else df['partContent']

    # to distill questions from the entire problem statement

    ans = []
    print("before clean_and_restore", datetime.now().strftime("%H:%M:%S"))

    cleaned_text = clean_and_restore_latex(str(df["total_text"]))
    # sentences = [sent for text in cleaned_text for sent in nltk.sent_tokenize(str(text))]
    print("after clean_and_restore", datetime.now().strftime("%H:%M:%S"))
    questions = nltk.sent_tokenize(cleaned_text)
    print("after tokenize", datetime.now().strftime("%H:%M:%S"))

    # Detect questions in the list of sentences
    ans = [sentence for sentence in questions if is_question(sentence)]

    if ans == []:
         ans = [sentence for sentence in questions if lenient_is_question(sentence)] 

    print("after isQuestion", datetime.now().strftime("%H:%M:%S"))

    # add the distilled questions back to the main table
    df["questionContent"] = ans
    # df["question_sentence_len"] = len(ans)


    #impute questionContent with partContent if not found.
    df['questionContent'] = [df['partContent']] if df["questionContent"] == [] else df['questionContent']

    print("before title vectornizer", datetime.now().strftime("%H:%M:%S"))
    # extract noun from title
    with open('title_noun_vectorizer.pkl', 'rb') as file:
        title_noun_vectorizer = pickle.load(file)
    # identified top20 terms with stronger correlation with label i.e >0.08 correlation and appeared
    # at least 5 times
    title_noun_terms = ["challenge","hydrostatic","tunnel", "strain","wind","cycle calculations","fourier","flow","fluid","stresses"]

    tokenized_doc = [extract_term_title(df['questiontitle'], title_noun_terms)]
    count_matrix = title_noun_vectorizer.transform(tokenized_doc)

    X_title_noun = count_matrix.toarray()

    print("after title vectornizerm  before sol vetornize", datetime.now().strftime("%H:%M:%S"))

    # extract verb, noun from solution

    with open('q_vectorizer.pkl', 'rb') as file:
        q_vectorizer = pickle.load(file)

    # sol_terms = ["image","equation","use","substitute","follow","apply","part","find","note","give","rearrange","diagram","look","stress"]
    # q_terms =[ 'delta', 'engine', 'find', 'frac', 'infty', 'left', 'omega', 'part', 'pressure', 'prime', 'river', 'shaft', 'show', 'text']
    q_terms =[ 'delta', 'engine', 'find', 'frac', 'infty', 'left', 'omega', 'part', 'pressure', 'prime', 'shaft', 'show', 'text']



    # Apply the extraction function to the DataFrame
    tokenized_doc = [extract_term_solution(str(text), q_terms) for text in df['total_text']]

    count_matrix = q_vectorizer.transform(tokenized_doc)

    X_q_noun = count_matrix.toarray()



    with open('sol_vectorizer.pkl', 'rb') as file:
        sol_vectorizer = pickle.load(file)

    sol_terms = ['align', 'array', 'cfrac', 'dfrac', 'dot', 'e_', 'frac', 'lambda', 'ldots', 'left', 'mathrm', 'omega', 'partial', 'ratio', 'right', 'sigma_', 'text', 'theta']


    # Apply the extraction function to the DataFrame
    tokenized_doc = [extract_term_solution(str(df['workedsolution']), sol_terms)]

    count_matrix = sol_vectorizer.transform(tokenized_doc)

    X_sol_noun = count_matrix.toarray()
    # to extract all question verb 
    print("after sol vectoriznt", datetime.now().strftime("%H:%M:%S"))


    # Extract nouns and verbs from the text data
    tokenized_doc = extract_question_verbs(str(df["questionContent"]))
    # processed_docs = [doc for doc in  df["questionContent"]]

    verb_count = len(tokenized_doc.split())
    df["verb_count_q"] = verb_count

    print("after verb count q, before token len count", datetime.now().strftime("%H:%M:%S"))

    tokenized_texts = custom_tokenize(str(df['total_text']))[0]
    tokenized_texts = [i for i in tokenized_texts if i!='']
    latex_terms = custom_tokenize(str(df['total_text']))[1]
    len_text_text = len(tokenized_texts)
    a =  element_counter(''.join(latex_terms))
    len_latex_text = len(a[1])
    df["text_latex_stats"] = a[0]

    # tokenized_solution = custom_tokenize(str(df['workedsolution']))[0]
    # tokenized_solution = [i for i in tokenized_solution if i!='']
    latex_terms_solution = custom_tokenize(str(df['workedsolution']))[1]
    # len_text_solution = len(tokenized_solution)
    a = element_counter(''.join(latex_terms_solution))
    len_latex_solution = len(a[1])
    df["solution_latex_stats"]  = a[0]

    tokenized_question = custom_tokenize(str(df['questionContent']))[0]
    tokenized_question = [i for i in tokenized_question if i!='']
    latex_terms_question = custom_tokenize(str(df['questionContent']))[1]
    len_text_question = len(tokenized_question)
    a = element_counter(''.join(latex_terms_question))
    len_latex_question = len(a[1])

    df["text_len"] = len_text_text
    df["latex_len"] = len_latex_text
    # df["latex_len_solution"] = len_latex_solution
    # df["text_len_solution"] = len_text_solution
    df["text_len_question"] = len_text_question
    df["latex_len_question"] = len_latex_question

    print("after token len count", datetime.now().strftime("%H:%M:%S"))


    ## add new features 
    df["char_len_total_text"] = len(str(df["total_text"]).replace(' ','').replace('\n',''))
    df["total_sol_len"] = len((str( df["workedsolution"])).replace(' ','').replace('\n',''))
    df["steps"] = str(df["workedsolution"]).count("***")
    if df["steps"] == 0 and df["total_sol_len"] != 0:
        # Count the number of colons
        steps = df["workedsolution"].count(":")
        # If still zero, count the number of double newlines
        if steps == 0:
            steps = len(re.findall(r'(\n\n)+', str(df["workedsolution"]))) // 2
        # If still zero and 'workedsolution' is not empty, set steps to 1
        if steps == 0:
            steps = 1

    df["skill_x_total_sol_len"] = df["skill"] * df["total_sol_len"] 
    # df["skill_x_steps"] = df["skill"] * df["steps"]
    # df["level_x_total_sol_len"] = df["total_sol_len"] * df["level"]

    # print("before embedding", datetime.now().strftime("%H:%M:%S"))
    
    # embeddings = model.encode(df["questionContent"])
    # print(embeddings.shape)
    # with open('knn_cluster.pkl', 'rb') as file:
    #     kmeans = pickle.load(file)
    # cluster_label = kmeans.predict(embeddings)
    # df["cluster"] = cluster_label[0]

    print("after embedding", datetime.now().strftime("%H:%M:%S"))

    included_fields = ['setnumber', 'questionnumber', 'skill', 'level', 'verb_count_q', 'text_latex_stats', 'solution_latex_stats', 'text_len', 'latex_len', 
                       'text_len_question', 'latex_len_question', 'char_len_total_text', 'total_sol_len', 'steps', 'skill_x_total_sol_len']
    # X_input = [X_sol_noun[0], X_title_noun[0]]
    # X_input.extend([ df[key] for key in included_fields])
    # X_input = np.array([X_sol_noun[0], X_title_noun[0]] + [ df[key] for key in included_fields])
    # X_input = np.array(X_input)

    X_input = np.concatenate([X_q_noun, X_sol_noun, X_title_noun, np.array([ [df[key]] for key in included_fields]).T  ], axis=1)
    print(X_input)
    with open('rf_classifier.pkl', 'rb') as file:
        rf_classifier = pickle.load(file)

    print("after rf_classify", datetime.now().strftime("%H:%M:%S"))

    bucket = rf_classifier.predict(X_input)[0]
    bucket = round(bucket)
    lower = max(bucket - 8,0)
    upper = bucket + 8
    return lower, upper
