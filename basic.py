from flask import Flask, render_template, request
import string
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from nltk.stem.wordnet import WordNetLemmatizer


lem = WordNetLemmatizer()
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/report')
def report():
    raw_text = request.args.get('raw_text')

    def text_process(mess):
        """
        1. remove punc
        2. remove stop words
        3. return list of clean text words
        """

        nopunc = ''.join([char for char in mess if char not in string.punctuation])
        lem_words = []
        for word in word_tokenize(nopunc):
            lem_words.append(lem.lemmatize(word))

        return [word for word in lem_words if word not in stopwords.words('english')]

    clean_text = text_process(raw_text)
    word_freq = FreqDist(clean_text)
    plt.ion()
    word_freq.plot(10, title='Top 10 Most common Words in Text')
    plt.savefig('static/images/plot.png')
    plt.ioff()



    return render_template('report.html', name='frequency distribution plot', url='static/images/plot.png')


if __name__ == '__main__':
    app.run(debug=True)
