import os
import pickle
import operator
from tqdm import tqdm
from Normalize import Normalize

class Train:
  def __init__(self, DOCS_DIR_PATH = 'IRTM/', CLASSIFIED_DOCS_PATH = 'training.txt'):
    self.DOCS_DIR_PATH = DOCS_DIR_PATH
    self.classfied_docs = {}

    with open(CLASSIFIED_DOCS_PATH, 'r') as f:
      for line in f:
        arr = line.rstrip().split(' ')
        c = int(arr[0]) # currentClass
        self.classfied_docs[c] = arr[1:]  # the other docs

  def CountDocs(self):
    return 195  # 195

  def CountDocsInClass(self, C, N):
    prior = {}

    for c in C:
      cLen = len(self.classfied_docs[c])
      prior[c] = cLen / N  # P(c)

    return prior
  # CountDocsInClass({1,2,3,4,5,6,7,8,9,10,11,12,13}, 1)

  def CountTokensOfTerm(self, corpus_arr, term): # corpus: 'aio, aopsd. sjdoasi...', term: 'a' / 'b' / ...
    count = 0
    for token in corpus_arr:
      if(token == term):
        count += 1

    return count
  # CountTokensOfTerm('aaSdijasiodasd asopdpop \' asiodjasoidj asijdoias asijdoias', '\'')

  def computeFeatureUtility(self, D_CONTENTS, t, C):  # D_CONTENT = { class: [ { p1's voc }, { p2's }, { p3's }, ... } ] }
    N = len(D_CONTENTS)
    total = 0

    for c in C:
      q = 0 # et=1, ec=1
      q_expect = 7.5
      w = 0 # et=0, ec=1
      w_expect = 7.5
      e = 0 # et=1, ec=0
      e_expect = 90
      r = 0 # et=0, ec=0
      r_expect = 90
      for singleArticleVoca in D_CONTENTS[c]:
        if(t in singleArticleVoca):
          q += 1
        else:
          w += 1
      for otherC in C.difference({ c }):
        for singleArticleVoca in D_CONTENTS[otherC]:
          if(t in singleArticleVoca):
            e += 1
          else:
            r += 1
      for (real, expect) in [(q, q_expect), (w, w_expect), (e, e_expect), (r, r_expect)]:
        total += ((real - expect) ** 2) / expect

    total /= len(C)

    if(q + e < 10): total = 0 # threshold: 10 times

    return total


  def kFeaturesWithLargestValues(self, Arr, k):  # Arr = [(#1, val1), (#2, val2), ...]
    s = sorted(Arr, key=operator.itemgetter(1))
    # print({(fea, val) for (fea, val) in s[-k:]})
    return {fea for (fea, val) in s[-k:]}

  def FeatureSelect(self, preFilteredVocabulary, D, C, k=500):
    features = {}
    filteredVocabulary = preFilteredVocabulary
    L = []
    D_CONTENTS = {}

    for c in tqdm(C, desc='2. Select top ' + str(k) + ' features'):
      D_CONTENTS[c] = []
      for docId in self.classfied_docs[c]:
        DOC_PATH = self.DOCS_DIR_PATH + str(docId) + '.txt'
        with open(DOC_PATH, 'r') as f:
          content = f.read()
          singleDocVocabulary = set(Normalize(content).process())
          D_CONTENTS[c].append(singleDocVocabulary)

    for t in tqdm(preFilteredVocabulary, desc='     Estimate & compate scores(threshold: appear 10 times)'):
      val = self.computeFeatureUtility(D_CONTENTS, t, C)
      L.append((t, val))
    filteredVocabulary = self.kFeaturesWithLargestValues(L, k)
    return filteredVocabulary

  def ExtractVocabulary(self, D):  # D: { 1, 2, 3, 4, 5, ..., N }
    preV = set()
    for docId in tqdm(D, desc='1. Establish the [vocabulary]'):
      DOC_PATH = self.DOCS_DIR_PATH + str(docId) + '.txt'
      with open(DOC_PATH, 'r') as f:
        content = f.read()
        singleDocVocabulary = Normalize(content).process()
      preV = preV.union(singleDocVocabulary)

    return preV
  # ExtractVocabulary({1,2,3,4})

  def ConcatenateTextOfAllDocsInClass(self, c): # c: 1 / 2 / ...
    targetDocIds = self.classfied_docs[c]
    totalDocContent = ''

    for docId in targetDocIds:
      DOC_PATH = self.DOCS_DIR_PATH + str(docId) + '.txt'
      with open(DOC_PATH, 'r') as f:
        rawDocContent = f.read()
        totalDocContent += rawDocContent
    
    return totalDocContent
  # ConcatenateTextOfAllDocsInClass(3)

  def TrainMultinomialNB(self, C, D): # C: {1, 2, 3, ..., 13}, D: { 1, 2, 3, 4, 5, ..., N } 
    preV = self.ExtractVocabulary(D)
    V = self.FeatureSelect(preV, D, C)
    N = self.CountDocs()
    prior = self.CountDocsInClass(C, N)
    condprob = {}
    firstTime = True  # to initialize condprob[t] to avoid weird mistakes
    for c in tqdm(C, desc='3. Calculate [prior] & [condition]'):
      T_total = 0  # total term counts in specific class
      text_c = self.ConcatenateTextOfAllDocsInClass(c)
      text_c_arr = Normalize(text_c).process()
      for t in V:
        T_ct = self.CountTokensOfTerm(text_c_arr, t)
        if(firstTime == True):
          condprob[t] = {}
        condprob[t][c] = T_ct
        T_total += T_ct
      for t in V:
        condprob[t][c] = (condprob[t][c] + 1) / (T_total + 1)  # P(X=t|c)
      firstTime = False

    return V, prior, condprob

if __name__ == "__main__":
  def gatherAllTrainingDocId(CLASSIFIED_DOCS_PATH = 'training.txt'):
    res = []
    with open(CLASSIFIED_DOCS_PATH, 'r') as f:
      for line in f:
        arr = line.rstrip().split(' ')
        res += arr[1:]
    return res

  pkg = Train().TrainMultinomialNB(C={ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 }, D=gatherAllTrainingDocId())
  print('\n================================ Save parameters in \'./V_prior_condprob.pkl\' ================================\n')

  with open('V_prior_condprob.pkl', 'wb') as handle:
    pickle.dump(pkg, handle)
  # aaa().TrainMultinomialNB({1,2,3}, { i+1 for i in range(100) })
