import numpy as np
import pandas as pd
import re
import jieba
from pprint import pprint
import jieba.posseg as pseg
import os
import sys
sys.stderr = open(os.devnull, "w")  # silence stderr
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import paddle

# Plotting tools
import matplotlib.pyplot as plt
def get_other_stopword_list(text_words):
    per_list = []  # 人名列表
    org_list = []
    time_list = []
    loc_list = []
    for word_list in text_words:
        for word in word_list:
            if len(word) == 1:
                continue
            words = pseg.cut(word, use_paddle=True)  # paddle模式
            # print(list(words))
            word, flag = list(words)[0]
            if flag == 'PER':
                if word not in per_list:
                    per_list.append(word)
            elif flag == 'ORG':
                if word not in org_list:
                    org_list.append(word)
            elif flag == 'TIME':
                if word not in time_list:
                    time_list.append(word)
            elif flag == 'LOC':
                if word not in loc_list:
                    loc_list.append(word)

    # print(per_list)
    return per_list, org_list, time_list, loc_list

def get_stop_words(path):
    return [item.strip() for item in open(path, 'r', encoding='utf-8').readlines()]

def drop_stopwords(line, stopwords):
    line_clean = []
    for word in line:
        if word in stopwords:
            continue
        line_clean.append(word)
    return line_clean
if __name__ == '__main__':
    paddle.enable_static()
    jieba.enable_paddle()
    sys.stderr = sys.__stderr__  # unsilence stderr
    stopword_path = "../data/stopword.txt"
    filepath = "../data/信访事项（1）.csv"
    data = pd.read_csv(filepath, encoding="GBK")
    texts = data["反映内容"].values.tolist()
    #去掉身份证号
    for index, text in enumerate(texts):
        text = re.sub('[\d|x]{3}', '', text)
        text = re.sub('[一|二|三|四|五|六|七|八|九|十|采油]+厂', '', text)
        texts[index] = text
    train_seg_text = [jieba.lcut(s) for s in texts]
    stopwords = get_stop_words(stopword_path)
    per_list, org_list, time_list, loc_list = get_other_stopword_list(train_seg_text)
    stopwords = list(set(stopwords + per_list + org_list + time_list + loc_list))
    train_st_text = [drop_stopwords(s, stopwords) for s in train_seg_text]
    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(train_st_text, min_count=5, threshold=100) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[train_st_text], threshold=100)

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]

    def make_trigrams(texts):
        return [trigram_mod[bigram_mod[doc]] for doc in texts]

    data_words_bigrams = make_bigrams(train_st_text)
    id2word = corpora.Dictionary(data_words_bigrams)     # Create Dictionary
    id2word.save_as_text("dictionary")                   # save dict
    texts = data_words_bigrams                           # Create Corpus
    corpus = [id2word.doc2bow(text) for text in texts]   # Term Document Frequency
    # lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
    #                                            id2word=id2word,
    #                                            num_topics=22,
    #                                            random_state=100,
    #                                            update_every=1,
    #                                            chunksize=100,
    #                                            passes=10,
    #                                            alpha='auto',
    #                                            per_word_topics=True)
    # # Print the Keyword in the 10 topics
    # pprint(lda_model.print_topics())
    # doc_lda = lda_model[corpus]
    # # Compute Perplexity
    # print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.
    #
    # # Compute Coherence Score
    # coherence_model_lda = CoherenceModel(model=lda_model, texts=data_words_bigrams, dictionary=id2word, coherence='c_v')
    # coherence_lda = coherence_model_lda.get_coherence()
    # print('\nCoherence Score: ', coherence_lda)   # 越高越好
    os.environ.update({'MALLET_HOME': r'D:\\xinfang\\data\\mallet-2.0.7'})
    mallet_path = r'D:\\xinfang\\data\\mallet-2.0.7\\bin\\mallet'
    # ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=22, id2word=id2word)
    # # Show Topics
    # pprint(ldamallet.show_topics(formatted=False))
    #
    # # Compute Coherence Score
    # coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=data_words_bigrams, dictionary=id2word,
    #                                            coherence='c_v')
    # coherence_ldamallet = coherence_model_ldamallet.get_coherence()
    # print('\nCoherence Score: ', coherence_ldamallet)


    # def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    #     """
    #     Compute c_v coherence for various number of topics
    #
    #     Parameters:
    #     ----------
    #     dictionary : Gensim dictionary
    #     corpus : Gensim corpus
    #     texts : List of input texts
    #     limit : Max num of topics
    #
    #     Returns:
    #     -------
    #     model_list : List of LDA topic models
    #     coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    #     """
    #     num_topics_list = []
    #     coherence_values = []
    #     for num_topics in range(start, limit, step):
    #         num_topics_list.append(num_topics)
    #         model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word)
    #         coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
    #         coherence_values.append(coherencemodel.get_coherence())
    #
    #     return  num_topics_list, coherence_values
    #
    #
    # num_topics_list, coherence_values = compute_coherence_values(
    #     dictionary=id2word, corpus=corpus, texts=data_words_bigrams, start=5, limit=30, step=5)
    #
    # # Show graph
    # plt.plot(num_topics_list, coherence_values)
    # plt.xlabel("Num Topics")
    # plt.ylabel("Coherence score")
    # plt.legend(("coherence_values"), loc='best')
    # plt.show()
    # # Print the coherence scores
    # for m, cv in zip(num_topics_list, coherence_values):
    #     print("Num Topics =", m, " has Coherence Value of", round(cv, 4))
    best_num_topics = 20
    # # Select the model and print the topics
    optimal_model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=best_num_topics, id2word=id2word)
    model_topics = optimal_model.show_topics(formatted=False)
    pprint(optimal_model.print_topics(num_words=10))
    coherencemodel = CoherenceModel(model=optimal_model, texts=texts, dictionary=id2word, coherence='c_v')
    print('\nCoherence Score: ', coherencemodel.get_coherence())


    def format_topics_sentences(ldamodel=optimal_model, corpus=corpus, texts=texts):
        # Init output
        sent_topics_df = pd.DataFrame()

        # Get main topic in each document
        for i, row in enumerate(ldamodel[corpus]):
            row = sorted(row, key=lambda x: (x[1]), reverse=True)
            # Get the Dominant topic, Perc Contribution and Keywords for each document
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:  # => dominant topic
                    wp = ldamodel.show_topic(topic_num)
                    topic_keywords = ", ".join([word for word, prop in wp])
                    sent_topics_df = sent_topics_df.append(
                        pd.Series([int(topic_num), round(prop_topic, 4), topic_keywords]), ignore_index=True)
                else:
                    break
        sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

        # Add original text to the end of the output
        contents = pd.Series(texts)
        sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
        return (sent_topics_df)


    df_topic_sents_keywords = format_topics_sentences(ldamodel=optimal_model, corpus=corpus, texts=data["反映内容"].values.tolist())

    # Format
    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

    # Show
    df_dominant_topic.to_csv('../result/lda_result.csv', encoding='utf-8')









