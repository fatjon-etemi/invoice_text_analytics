import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


class Processer():
    def fit(self, X, Y=None):
        return self

    def transform(self, X, Y=None):
        tokens_list = []
        for doc in X:
            tokens = nltk.word_tokenize(doc)
            stop_words = stopwords.words('english') + stopwords.words('german')
            tokens = [x for x in tokens if not x in stop_words]
            tokens = [x.lower() for x in tokens]
            tokens = [x for x in tokens if x.isalpha()]
            porter = PorterStemmer()
            tokens = [porter.stem(x) for x in tokens]
            tokens_list.append(tokens)
        return tokens_list

    def fit_transform(self, X, Y=None):
        self.fit(X, Y)
        return self.transform(X, Y)
