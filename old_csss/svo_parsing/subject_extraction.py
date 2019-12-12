
import os
import re
import pickle
import nltk
import numpy

stop = nltk.corpus.stopwords.words('english')

class SubjectTrigramTagger(object):

    """ Creates an instance of NLTKs TrigramTagger with a backoff
    tagger of a bigram tagger a unigram tagger and a default tagger that sets
    all words to nouns (NN)
    """

    def __init__(self, train_sents):

        """
        train_sents: trained sentences which have already been tagged.
                Currently using Brown, conll2000, and TreeBank corpuses
        """
        #frac = int(numpy.round(0.8*len(train_sents)))
        t0 = nltk.DefaultTagger('NN')
        t1 = nltk.UnigramTagger(train_sents, backoff=t0)
        t2 = nltk.BigramTagger(train_sents, backoff=t1)
        self.tagger = nltk.TrigramTagger(train_sents, backoff=t2)

        #print(self.tagger.evaluate(train_sents[frac:]))

    def tag(self, tokens):
        return self.tagger.tag(tokens)

# Noun Part of Speech Tags used by NLTK
# More can be found here
# http://www.winwaed.com/blog/2011/11/08/part-of-speech-tags/
NOUNS = ['NN', 'NNS', 'NNP', 'NNPS']
VERBS = ['VB', 'VBG', 'VBD', 'VBN', 'VBP', 'VBZ',
         'BE', 'BEG', 'BEM', 'BER', 'BEZ', 'BEN', 'BED', 'BEDZ',
         'HV', 'HVD', 'HVG', 'HVN', 'HVZ',
         'DO', 'DID', 'DOES'
        ]

def clean_document(document):
    """Remove enronious characters. Extra whitespace and stop words"""
    document = re.sub('[^A-Za-z0-9\’\' .-]+', ' ', document)
    document = ' '.join(document.split())
    # document = ' '.join([i for i in document.split() if i not in stop])
    return document

def tokenize_sentences(document):
    sentences = nltk.sent_tokenize(document)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    return sentences

def get_entities(document):
    """Returns Named Entities using NLTK Chunking"""
    entities = []
    sentences = tokenize_sentences(document)

    # Part of Speech Tagging
    sentences = [nltk.pos_tag(sent) for sent in sentences]
    for tagged_sentence in sentences:
        for chunk in nltk.ne_chunk(tagged_sentence):
            if type(chunk) == nltk.tree.Tree:
                entities.append(' '.join([c[0] for c in chunk]).lower())
    return entities

def extract_subjects(document):
    # Get most frequent Nouns (excluding stopwords)
    words = nltk.tokenize.word_tokenize(document)
    fdist = nltk.FreqDist([word.lower() for word in words if (word not in stop) and (word != '’')])
    most_freq_nouns = [w for w, c in fdist.items()
                       if nltk.pos_tag([w])[0][1] in NOUNS]

    # Get Top 10 entities
    entities = get_entities(document)
    most_freq_entities = [w for w, c in nltk.FreqDist(entities).items()]

    # Return all nouns and named entities
    return most_freq_nouns + most_freq_entities

def extract_verbs(document):
    # Get most frequent Verbs (excluding stopwords)
    words = nltk.tokenize.word_tokenize(document)
    fdist = nltk.FreqDist([word.lower() for word in words if word not in stop])
    most_freq_verbs = [w for w, c in fdist.items()
                       if nltk.pos_tag([w])[0][1] in VERBS]

    # Return all verbs
    return most_freq_verbs

def trained_tagger(existing=True):
    """Returns a trained trigram tagger

    existing : set to True if already trained tagger has been pickled
    """
    current_loc = os.path.dirname(__file__)
    if existing:
        trigram_tagger = pickle.load(open(os.path.join(current_loc, 'trained_tagger.pkl'), 'rb'))
        return trigram_tagger

    # Aggregate trained sentences for N-Gram Taggers
    train_sents = nltk.corpus.brown.tagged_sents()
    train_sents += nltk.corpus.conll2000.tagged_sents()
    train_sents += nltk.corpus.treebank.tagged_sents()

    # Create instance of SubjectTrigramTagger and persist instance of it
    trigram_tagger = SubjectTrigramTagger(train_sents)
    pickle.dump(trigram_tagger, open(os.path.join(current_loc, 'trained_tagger.pkl'), 'wb'))

    return trigram_tagger

def tag_sentences(subject, document):
    """Returns tagged sentences using POS tagging"""
    trigram_tagger = trained_tagger()

    # Tokenize Sentences and words
    sentences = tokenize_sentences(document)
    sentences = merge_multi_word_subject(sentences, subject)

    # Filter out sentences where subject is not present
    sentences = [sentence for sentence in sentences if subject in
                [word.lower() for word in sentence]]

    # Tag each sentence
    tagged_sents = [trigram_tagger.tag(sent) for sent in sentences]
    return tagged_sents

def merge_multi_word_subject(sentences, subject):
    """Merges multi word subjects into one single token
    ex. [('steve', 'NN', ('jobs', 'NN')] -> [('steve jobs', 'NN')]
    """
    if len(subject.split()) == 1:
        return sentences
    subject_lst = subject.split()
    sentences_lower = [[word.lower() for word in sentence]
                        for sentence in sentences]
    for i, sent in enumerate(sentences_lower):
        if subject_lst[0] in sent:
            for j, token in enumerate(sent):
                start = subject_lst[0] == token
                exists = subject_lst == sent[j:j+len(subject_lst)]
                if start and exists:
                    del sentences[i][j+1:j+len(subject_lst)]
                    sentences[i][j] = subject
    return sentences

def get_svos(sentence, subject):
    """Returns a list of dictionaries containing:

    subject : the subject determined earlier
    action : the action verb of particular related to the subject
    object : the object the action is referring to
    phrase : list of token, tag pairs for that lie within the indexes of
                the variables above
    """
    subject_idx = next((i for i, v in enumerate(sentence)
                    if v[0].lower() == subject), None)
    data = [] 
    for i in range(subject_idx, len(sentence)):
        entry = {'subject': subject}
        found_action = False
        for j, (token, tag) in enumerate(sentence[i+1:]):
            if token == "’": 
                continue
            if tag in VERBS:
                entry['action'] = token
                found_action = True
            if tag in NOUNS and found_action == True:
                entry['object'] = token
                entry['phrase'] = sentence[i: i+j+2]
                
                # Append the SVO
                data.append(entry)

                # Continue looking for others
                entry = {'subject': subject}
                found_action = False

            #print(subject, token, tag, found_action, len(data))
    return data
    