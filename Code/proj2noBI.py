import re
import numpy as np
from collections import Counter
import csv
import operator
import sklearn.linear_model
import scipy as sp
#from sklearn.linear_model import Lasso
import re
import gensim
import gensim.models


""" given a file path returns a list of all so 'lines' in the .txt file """
def clean(file_path):
    opn = open(file_path.format(0)).read()
    opn = opn.replace("\n"," ")
    opn = opn.replace("’ ","’")
    return opn.split(" ")

def delBI(lst):
    newlst = []
    for i in range(len(lst)):
        if lst[i] != 'O':
            newlst.append(lst[i][2:])
        else:
            newlst.append(lst[i])
    return newlst

""" call clean function with training data """
train = clean("Project2_fall2018/train_cut.txt")
dev = clean("Project2_fall2018/dev.txt")



""" create separate lists of tokens, pos tags and bio tags """
tokens_train = []
pos_tags_train = []
bio_tags_train = []
for i in range(len(train)):
    if i % 3 == 0:
        individs = train[i].split("\t")
        tokens_train.extend(individs)
    elif i % 3 == 1:
        individs = train[i].split("\t")
        pos_tags_train.extend(individs)
    elif i % 3 == 2:
        individs = train[i].split("\t")
        bio_tags_train.extend(individs)

tokens_train = tokens_train[:-1]
bio_tags_train = delBI(bio_tags_train)

""" create separate lists of tokens, pos tags and bio tags """
tokens_dev = []
pos_tags_dev = []
bio_tags_dev = []
for i in range(len(dev)):
    if i % 3 == 0:
        individs = dev[i].split("\t")
        tokens_dev.extend(individs)
    elif i % 3 == 1:
        individs = dev[i].split("\t")
        pos_tags_dev.extend(individs)
    elif i % 3 == 2:
        individs = dev[i].split("\t")
        bio_tags_dev.extend(individs)

bio_tags_dev = delBI(bio_tags_dev)

#""" Seprating training data and development set. """
#tokens_train = tokens[0:200000]
#tokens_dev = tokens[200000:]
#pos_tags_train = pos_tags[0:200000]
#pos_tags_dev = pos_tags[200000:]
#bio_tags_train = bio_tags[0:200000]
#bio_tags_dev = bio_tags[200000:]

""" receives list of tokens or tags and returns dictionary with unique tokens or tags
    as keys and unigram frequency of tokens or tags throughout the text as values"""
def getUniCount(lst):
    countdict = dict()
    for word in lst:
        if word in countdict:
            countdict[word] += 1
        else:
            countdict[word] = 1
    return countdict

""" receives a unicount dictionary and length of  returns dictionary with unique tokens or tags
    as keys and unigram probabioities of tokens or tags occuring in whole doc as values"""
def getUniProb(count_dict):
    length = sum(count_dict.values())
    probdict = dict()
    for k,v in count_dict.items():
        probdict[k] = v/length
    return probdict

""" receives a list of tokens or tags and returns a 2D dictionary for bigram frequencies"""
def getBiCount(lst):
    countdict = dict()
    for i in range(len(lst)-1):
        curr = lst[i]
        nxt = lst[i+1]
        if curr in countdict.keys():
            if nxt in countdict[curr]:
                countdict[curr][nxt] += 1
            else:
                countdict[curr][nxt] = 1
        else:
            countdict[curr] = {nxt: 1}
    return countdict

""" receives a list of tokens or tags and returns a 2D dictionary for bigram probabioities"""
def getBiProb(unicount_dict,bicount_dict):
    probdict = dict()
    for k1,v1 in bicount_dict.items():
        probdict[k1] = dict()
        for k2,v2 in v1.items():
            probdict[k1][k2] = v2 / unicount_dict[k1]
    return probdict


""" receives list of tokens and returns a 2D dictionary for bigram frequencies"""
def getTagTokCount(tgs,tkns):
    countdict = dict()
    for i in range(len(tkns)-1):
        tag = tgs[i]
        word = tkns[i]
        if tag in countdict.keys():
            if word in countdict[tag].keys():
                countdict[tag][word] += 1
            else:
                countdict[tag][word] = 1
        else:
            countdict[tag] = {word: 1}
    return countdict

""" receives a list of unique tokens and a bigram count dictionary (word | tag)
    returns a new 2D dictionary that maps word -> tag_i -> frequency """
def wordToTagDict(unique_tokens, bicount):
    newdict = dict()
    for word in unique_tokens:
        newdict[word] = dict()
        for tag in bicount.keys():
            if word in bicount[tag].keys():
                newdict[word][tag] = bicount[tag][word]
            else:
                newdict[word][tag] = 0
    return newdict


""" receives a list of tokens and a unicount dict for the tokens,
    and returns a new list of tokens with tokens with only one count
    substituted to a <unk> tag"""
def unknown1(token_lst, unicount_dict):
    newlist = []
    for word in token_lst:
        if unicount_dict[word] == 1:
            newlist.append("<unk>")
        else:
            newlist.append(word)
    return newlist


""" receives a word given tag count dictionary (emission count) and
    a tag count dictionary.
    returns a 2D probabioity dictionary for word given tag with
    tag as the first level key and word as the second level key. """
def emProb(em_count_dict, tag_count_dict):
    newdict = dict()
    for tag,v1 in em_count_dict.items():
        newdict[tag] = dict()
        for word,count in v1.items():
            newdict[tag][word] = count / tag_count_dict[tag]
    return newdict

""" receives a word given tag count dictionary (emission count) and
    a tag count dictionary.
    returns a 2D dict with emission probabioities smoothed using
    Laplace Smoothing method """
def laplaceSmoothing(em_count_dict, tag_count_dict):
    newdict = dict()
    for tag,v1 in em_count_dict.items():
        newdict[tag] = dict()
        for word,count in v1.items():
            tagSize = len(em_count_dict[tag].keys())
            newdict[tag][word] = (count + 1) / (tag_count_dict[tag] + tagSize)
    return newdict

""" receives ...
    returns a 2D dict with emission probabioities smoothed using
    Laplace Smoothing method """
def addKSmoothing(em_count_dict, tag_count_dict, k):
    newdict = dict()
    for tag,v1 in em_count_dict.items():
        newdict[tag] = dict()
        for word,count in v1.items():
            tagSize = len(em_count_dict[tag].keys())
            newdict[tag][word] = (count + k) / (tag_count_dict[tag] + k*tagSize)
    return newdict


""" receives a unicount dictionary, a number k,
    and performs add k smoothing on the probabilities.
    returns a new dictionary with the smoothed probabilities. """
def smoothed_unigram(unicount, k):
    newdict = dict()
    unique_size = len(unicount.keys())
    data_length = sum(unicount.values())
    for w in unicount.keys():
        newdict[w] = (unicount[w] + k) / (data_length + k * unique_size)
    return newdict

""" receives list of float values and returns new list of float
    values summing to 1. fixes numpy random.choice glitch to ensure
    weighted probabioities sum to 1"""
def div_sum(vals):
    newlst = []
    sv = sum(vals)
    for v in vals:
        newlst.append(v / sv)
    return newlst


def viterbi(trans_prob, em_prob, tag_uniprob, word_seq):
    distinct_tags = tag_uniprob.keys()
    score = dict()
    bptr = dict()
    #initialization
    for tag in distinct_tags:
        score[tag] = dict()
        if word_seq[0] in em_prob[tag].keys():
            score[tag][0] = tag_uniprob[tag] * em_prob[tag][word_seq[0]]
        elif "<unk>" in em_prob[tag].keys():
            score[tag][0] = tag_uniprob[tag] * em_prob[tag]["<unk>"]
        else:
            #no uknown for this tag
            score[tag][0] = 0
        bptr[tag] = dict()
        bptr[tag][0] = "START"
    #iteration
    for i in range(len(word_seq)):
        if i != 0:
            for tag in distinct_tags:
                maxx = 0
                maxxtag = tag
                for prevtag in distinct_tags:
                    transp = 0
                    if prevtag in trans_prob.keys() and tag in trans_prob[prevtag].keys():
                        transp = trans_prob[prevtag][tag] #note: we do not smooth trans prob
                    if score[prevtag][i-1] * transp > maxx:
                        maxx = score[prevtag][i-1] * transp
                        maxxtag = prevtag
                if word_seq[i] in em_prob[tag].keys():
                    score[tag][i] = maxx * em_prob[tag][word_seq[i]]
                elif "<unk>" in em_prob[tag].keys():
                    score[tag][i] = maxx * em_prob[tag]["<unk>"]
                else:
                    score[tag][i] = 0
                bptr[tag][i] = maxxtag
    #return
    maxscore = 0
    maxtag = "O"
    for tag in distinct_tags:
        if (score[tag][len(word_seq) - 1]) >= maxscore:
            maxscore = score[tag][len(word_seq) - 1]
            maxtag = tag
    returnlst = []
    returnlst.append(maxtag)
    currback = bptr[maxtag][len(word_seq) - 1]
    for i in range(len(word_seq)):
        if i != 0:
            ind = len(word_seq) - i - 1
            returnlst.append(currback)
            currback = bptr[currback][ind]
    returnlst.reverse()
    return returnlst

def MEMMhelper(word, prev_bio_tag, bio_tag, pos_tag, weight_matrix, emCount, tagSeq, lexicon):
    new_sum = 0
    feature_arr = runFeats(word, pos_tag, prev_bio_tag, bio_tag, emCount, lexicon)
    for i in range(20):
        tagindex = tagSeq.index(bio_tag)
        w_i = weight_matrix[tagindex][i]
        f_i = feature_arr[i]
        internal_product = w_i * f_i
        new_sum += internal_product
    return np.exp(new_sum)

def viterbiMEMM(word_seq, pos_seq, weights2D, unique_tags, emCount, trans_prob, lexicon):
    score = dict()
    bptr = dict()
    #initialization
    for tag in unique_tags:
        score[tag] = dict()
        bptr[tag] = dict()
        score[tag][0] = MEMMhelper(word_seq[0], "", tag, pos_seq[0], weights2D, emCount, unique_tags, lexicon)
        #score[tag][0] = trans_prob_uni[tag]
        bptr[tag][0] = "START"
    #iteration
    for i in range(len(word_seq)):
        if (i != 0):
            node_arr = [] #2D array -> node_arr[tag][prevtag] = score(prevtag) * p(tag | word, prevtag)
            z_arr = [] #1D array -> z_arr[prevtag] = zsum  where zsum is sigma_tag p(tag | word, prevtag) for each prevtag
            #initialize node_arr
            for x in range(len(unique_tags)):
                new = []
                for y in range(len(unique_tags)):
                    new.append(0)
                node_arr.append(new)
            for pi in range(len(unique_tags)):
                prevtag = unique_tags[pi]
                zarray = [] #1D array -> zarray[tag] = p(tag | word, prevtag)
                for ti in range(len(unique_tags)):
                    tag = unique_tags[ti]
                    p = MEMMhelper(word_seq[i], prevtag, tag, pos_seq[i], weights2D, emCount, unique_tags, lexicon)
                    zarray.append(p)
                    node_arr[ti][pi] = p #* transp  #score[prevtag][i-1] *
                z_arr.append(np.sum(zarray))
            finalscore = []
            debug = 0
            for ti in range(len(unique_tags)):
                new = []
                for pi in range(len(unique_tags)):
                    prevtag = unique_tags[pi]
                    #final score should be node_arr[tag][prevtag] / z_arr[prevtag]
                    transp = 0
                    if prevtag in trans_prob.keys() and tag in trans_prob[prevtag].keys():
                        transp = trans_prob[prevtag][tag]
                    new.append(node_arr[ti][pi] * score[prevtag][i-1] *transp / z_arr[pi])
                    if pi == 0: debug += node_arr[ti][pi] / z_arr[pi]
                finalscore.append(new)
            for ti in range(len(unique_tags)):
                tag = unique_tags[ti]
                for pi in range(len(unique_tags)):
                    prev = unique_tags[pi]
                    score[tag][i] = np.amax(finalscore[ti])
                    bptr[tag][i] = unique_tags[np.argmax(finalscore[ti])]
                    # #get max by dummy way
                    # maxx = 0
                    # for z in range(len(finalscore[ti])):
                    #     if finalscore[ti][z] >= maxx:
                    #         maxx = finalscore[ti][z]
                    #         score[tag][i] = finalscore[ti][z]
                    #         bptr[tag][i] = unique_tags[z]

    #return
    maxscore = float("-inf")
    maxtag = "MISC"
    for tag in unique_tags:
        if (score[tag][len(word_seq) -1] >= maxscore):
            maxscore = score[tag][len(word_seq) -1]
            maxtag = tag
    returnlst = []
    returnlst.append(maxtag)
    currback = bptr[maxtag][len(word_seq) - 1]
    for i in range(len(word_seq)):
        if i != 0:
            ind = len(word_seq) - i - 1
            returnlst.append(currback)
            currback = bptr[currback][ind]
    returnlst.reverse()
    return returnlst



def runFeats(word,tag,bio1,bio2,tagtok_2d,lexicon):
    bios = ["B-LOC","B-ORG","B-PER","B-MISC","I-LOC","I-ORG","I-PER","I-MISC","O"]
    bios = ["LOC", "ORG", "PER", "MISC", "O"]
    temp = []
    # feature 1: NNP
    if tag == "NNP": temp.append(1)
    else: temp.append(0)
    # feature 2: NNPS
    if tag == "NNPS": temp.append(1)
    else: temp.append(0)
    # feature 3: ALL CAPS & length < 5
    if word.isupper() and len(word) < 5: temp.append(1)
    else: temp.append(0)
    # feature 4: Capitalization
    if word[0].isupper() and word[-1].islower(): temp.append(1)
    else: temp.append(0)

    #features 5-12: "B-LOC","B-ORG","B-PER","B-MISC","I-LOC","I-ORG","I-PER","I-MISC"
    for b in bios:
        if word in tagtok_2d[b].keys():
            num = tagtok_2d[b][word]
        elif "<unk>" in tagtok_2d[b].keys():
            num = tagtok_2d[b]["<unk>"]
        else:
            num = 0
        if word in tokens_unk_unicount:
            denom = tokens_unk_unicount[word]
        else:
            denom = tokens_unk_unicount["<unk>"]
        temp.append(num/denom)
    # feature 13: tag transitionary probability
    if bio1 == "": temp.append(bio_tags_uniprob[bio2])
    else:
        if bio1 in bio_tags_transprob.keys() and bio2 in bio_tags_transprob[bio1].keys():
            temp.append(bio_tags_transprob[bio1][bio2])
        else:
            temp.append(0)
    #feature 14-21: previous BIO tags
    for b in bios:
        if bio1 == b:
            temp.append(1)
        else: temp.append(0)
    #feature 22-29: lexicon (most frequent tag)
    most_freq_tag = "O"
    if word in lexicon.keys():
        # get the inner dictionary's value, which is an inner dict of
        # tag : freq
        inner_word_dict = lexicon[word]
        #get tag with max frequency
        most_freq_tag = max(inner_word_dict.items(), key=operator.itemgetter(1))[0]
    for b in bios:
        if most_freq_tag == b:
            temp.append(1)
        else: temp.append(0)

    # #feature 30: emphasize on previous B tag
    # if bio1[:1] == "B":
    #     temp.append(1)
    # else:
    #     temp.append(0)
    # #feature 31: emphasize on previous I tag
    # if bio1[:1] == "I":
    #     temp.append(1)
    # else:
    #     temp.append(0)
    return temp

""" takes in a unigram and bigram tag / word dictionary and runs backoff. returns new dictionary. """
def backoff(unigram,bigram):
    boff = bigram
    for tag in bigram.keys():
        for word in unigram.keys():
            if word not in bigram[tag].keys():
                boff[tag][word] = unigram[word]
    return boff

""" receives output and answer key and calculates PRF. returns F. """
def calcPRF(output,answer):
    correct_p = 0
    guesses_p = len(output)
    correct_r = 0
    key_r = 0
    for i in range(len(output)):
        o1 = output[i]
        a1 = answer[i]
        if o1 == a1:
            correct_p += 1
            if o1 != 'O':
                correct_r += 1
        if a1 != 'O':
            key_r += 1
    p = correct_p / guesses_p
    r = correct_r / key_r
    f = (2 * p * r) / (p + r)
    print("P: ")
    print(p)
    print("R: ")
    print(r)
    print("F: ")
    print(f)
    return f

def baseline(sentences, lexicon):
    ret_lst = []
    for sent in sentences:
        for word in sent:
            if word in lexicon.keys():
                # get the inner dictionary's value, which is an inner dict of
                # tag : freq
                inner_word_dict = lexicon[word]
                #get tag with max frequency
                tag = max(inner_word_dict.items(), key=operator.itemgetter(1))[0]
                ret_lst.append(tag)
            else:
                ret_lst.append("O")
    return ret_lst

""" given a file path returns a list of all so 'lines' in the .txt file """
def formatSents(file_path):
    opn = open(file_path.format(0)).read()
    #opn = opn.replace("\n"," ")
    opn = opn.replace("’ ","’")
    opn_lst = opn.split("\n")
    fin_lst = []
    for x in opn_lst:
        fin_lst.append(x.split("\t"))
    return fin_lst

""" call clean function with test data """
test_f = formatSents("Project2_fall2018/test.txt")
dev_f = formatSents("Project2_fall2018/dev.txt")

""" create separate lists of tokens, pos tags and bio tags for test set """
test_sents = []
test_pos_tags = []
test_index = []
for i in range(len(test_f)-1):
    if i % 3 == 0:
        test_sents.append(test_f[i])
    elif i % 3 == 1:
        test_pos_tags.append(test_f[i])
    elif i % 3 == 2:
        test_index.append(test_f[i])

""" create separate lists of tokens, pos tags and bio tags for dev set """
dev_sents = []
dev_pos_tags_ll = []
dev_bio_tags_ll = []
for i in range(len(dev_f)-1):
    if i % 3 == 0:
        dev_sents.append(dev_f[i])
    elif i % 3 == 1:
        dev_pos_tags_ll.append(dev_f[i])
    elif i % 3 == 2:
        dev_bio_tags_ll.append(delBI(dev_f[i]))


def viterbify(btt, bpus, btul, sents, memm):
    all_tags = []
    for sent in sents:
        tgs = viterbi(btt, bpus, btul, sent)
        all_tags.extend(tgs)
    return all_tags

def viterbifyMEMM(sents, postags, weights, unique_tags, emission_count, trans_prob, lexicon):
    all_tags = []
    for i in range(len(sents)):
        tgs = viterbiMEMM(sents[i], postags[i], weights, unique_tags, emission_count, trans_prob, lexicon)
        all_tags.extend(tgs)
    return all_tags

def csv_helper(lst):
    strng = ""
    inseq = False
    start = 0
    counter = 0
    for i in range(len(lst)):
        if i+1 == len(lst):
            if inseq:
                return strng + str(start) + "-" + str(lst[i])
            else:
                return strng + str(lst[i]) + "-" + str(lst[i])
        if lst[i] == (lst[i+1]-1):
            if not inseq:
                start = lst[i]
                inseq = True
            counter += 1
        else:
            if inseq:
                strng += str(start) + "-" + str(start+counter) + " "
                start = 0
                counter = 0
                inseq = False
            else:
                strng += str(lst[i]) + "-" + str(lst[i]) + " "

def makecsv(csv_file_name,lst):
    with open(csv_file_name + ".csv", 'w', newline='') as myfile:
        wr = csv.writer(myfile, delimiter=',', quoting=csv.QUOTE_ALL)
        wr.writerow(['Type', 'Prediction'])
        pers = []
        locs = []
        orgs = []
        miscs = []
        for i in range(len(lst)):
            if lst[i][-3:] == "PER":
                pers.append(i)
            elif lst[i][-3:] == "LOC":
                locs.append(i)
            elif lst[i][-3:] == "ORG":
                orgs.append(i)
            elif lst[i][-4:] == "MISC":
                miscs.append(i)

        wr.writerow(["PER", csv_helper(pers)])
        wr.writerow(["LOC", csv_helper(locs)])
        wr.writerow(["ORG", csv_helper(orgs)])
        wr.writerow(["MISC", csv_helper(miscs)])


def makeX(emCount, lexicon):
    arr = []
    for i in range(len(tokens_train_unk)):
        temp = runFeats(tokens_train_unk[i],pos_tags_train[i],"",bio_tags_train[i], emCount, lexicon)
        arr.append(temp)
    return arr


#training data set
bio_tags_unicount = getUniCount(bio_tags_train)
bio_tags_uniprob = getUniProb(bio_tags_unicount)
bio_tags_bicount = getBiCount(bio_tags_train)
bio_tags_transprob = getBiProb(bio_tags_unicount, bio_tags_bicount)

tokens_unicount = getUniCount(tokens_train)
tokens_train_unk = unknown1(tokens_train, tokens_unicount) #a list of tokens with some as unknown
tokens_unk_unicount = getUniCount(tokens_train_unk)
tokens_unk_uniprob = getUniProb(tokens_unk_unicount)
#tokens_uniprob = getUniProb(tokens_unicount)

emission_count = getTagTokCount(bio_tags_train, tokens_train)
emission_prob = emProb(emission_count, bio_tags_unicount)
emission_count_unk = getTagTokCount(bio_tags_train, tokens_train_unk)#unk
#emission_prob_unk = emProb(emission_count_unk, bio_tags_unicount) #unk
emission_prob_unk_smooth = addKSmoothing(emission_count_unk, bio_tags_unicount, 0.87) #unk with smoothing

tokens_uniprob_smooth = smoothed_unigram(tokens_unk_unicount, 0.8) #unk and smoothed
#backoff_prob = backoff(tokens_uniprob, emission_prob)
#backoff_prob_unk = backoff(tokens_unk_uniprob, emission_prob_unk)#unk
backoff_prob_unk_smooth = backoff(tokens_uniprob_smooth, emission_prob_unk_smooth) #unk with smoothing, might want to smooth uniprob
#
# #development set
# #HMM
# dev_guess = viterbify(bio_tags_transprob, backoff_prob_unk_smooth, bio_tags_uniprob, dev_sents)
# dev_PRF = calcPRF(dev_guess, bio_tags_dev)
#
#
# #baseline
lexicon = wordToTagDict(tokens_unk_unicount.keys(), emission_count_unk)
# dev_bs_guess = baseline(dev_sents, lexicon)
# dev_bs_PRF = calcPRF(dev_bs_guess, bio_tags_dev)
#
# #test
# test_guess = viterbify(bio_tags_transprob, backoff_prob_unk_smooth, bio_tags_uniprob, test_sents)
# #test_guess_baseline = baseline(test_sents, wordToTagDict(tokens_unk_unicount.keys(), emission_count_unk))
# makecsv("8780hmm",test_guess)
# #makecsv("baseline", test_guess_baseline)

#dev on MEMM
x2d = np.array(makeX(emission_count_unk, lexicon))
lr = sklearn.linear_model.LogisticRegression()
#lr = Lasso(positive=True)
lrresult = lr.fit(x2d, bio_tags_train)
weights = lrresult.coef_
unique_tags = np.ndarray.tolist(lrresult.classes_)
# print(weights[0])
# print(sum(weights[0]))
MEMMguess = viterbifyMEMM(dev_sents[0:5], dev_pos_tags_ll[0:5], weights, unique_tags, emission_count_unk, bio_tags_transprob, lexicon)
print(MEMMguess)
print("answer")
print(dev_bio_tags_ll[0:5])
calcPRF(MEMMguess, bio_tags_dev[0:(len(MEMMguess))])

# #actual test file for MEMM
# MEMMtest = viterbifyMEMM(test_sents, test_pos_tags, weights, unique_tags, emission_count_unk, bio_tags_transprob, lexicon)
# makecsv("testMEMM", MEMMtest)




def greedy(word_seq, pos_seq, weights2D, unique_tags, emCount):
    taglst = []
    for i in range(len(word_seq)):
        maxx = float("-inf")
        maxxtag = "B-PER"
        for tag in unique_tags:
            p = MEMMhelper(word_seq[i], "B-LOC", tag, pos_seq[i], weights2D, emCount, unique_tags)
            if (p >= maxx):
                maxx = p
                maxxtag = tag
        taglst.append(maxxtag)
    return taglst

# MEMMtrainguess = greedy(tokens_train_unk[0:100], pos_tags_train[0:100], weights, unique_tags, emission_count_unk)
# print(MEMMtrainguess[0:100])
# print(bio_tags_train[0:100])

def usePredict(tokenslst, postagslst):
    final = []
    for i in range(len(tokenslst)):
        arr = runFeats(tokenslst[i], postagslst[i], "", "", emission_count_unk)
        final.append(arr)
    return final

# print(lr.predict(final))
# print(bio_tags_train[0:100])
#
# print(lr.predict(usePredict(dev_sents[0], dev_pos_tags_ll[0])))
# print(dev_bio_tags_ll[0])
