# SentiGraph: a graph-based approach for sentiment analysis.

A sentiment analysis engine built in Python utilizing the EmoLex lexicon.

![Main UI](https://i.ibb.co/kGcPcvd/Screen-Shot-2020-07-13-at-8-29-43-PM.png)

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them:

```
1. NetworkX - Python graph building software (pip3 install NetworkX).
2. Gensim - Word2Vec models (pip3 install gensim).
3. MatPlotLib - Basic graph visualization software (pip3 install matplotlib).
4. NLTK - Python NLP library. (pip3 install nltk)
```

It is recommended that you use Python3 as it was developed using this version.

### Installing

Once you have cloned the repository, you need to setup the output directory for the graphs. Create a folder called 'graph_outputs' and inside that another folder 'v3' (this will hold the output of the graph). The graph is generated as a .pickle python output which converts the NetworkX model object into a storable bytes.

you can create the SentiGraph by running:

```
python3 emolex_graph.py
```

This will take around 20-30 minutes to create, so follow the on-screen dialogs for progress; modify the internal variables as you see fit including the output file location for the graph etc.

## Classifying Sentiment

Included are three files relating to anger, sadness, and joy. You can analyse any passage you like by specifying them in classifier.py and then classify by running:

```
python3 classifier.py
```

### Validation Testing

To run the tests as documented in the paper, make sure you have the associated validation test passages inside your testing folder; then run:

```
python3 testing.py
```

Which should return a classification report and associated confusion matrix.

## Built With

* [NLTK](https://www.nltk.org/) - NTLK Natural Language Processing Library for Python
* [EmoLex](https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm) - EmoLex Lexicon containing 14,000+ words.
* [Sklearn](https://scikit-learn.org/stable/) - Scikit-Learn for Python

## Authors

* **John McCambridge** - *Initial work*

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

I would like to give thanks to Prof. Rolando A. Coto Solano of Dartmouth College for his strong guidance and informative criticism during the pursuit of this project.
