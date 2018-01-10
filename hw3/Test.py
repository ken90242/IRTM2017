import sys
import pickle
from math import log
from tqdm import tqdm
from Normalize import Normalize

class SingleDoc:
  C = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 }
  V, prior, condprob = pickle.load(open('V_prior_condprob.pkl', 'rb'))
  DOCS_DIR_PATH = 'IRTM/'

  def __init__(self, docId):
    DOC_PATH = self.DOCS_DIR_PATH + str(docId) + '.txt'
    with open(DOC_PATH, 'r') as f:
      content = f.read()
#       singleDocVocabulary = Normalize(content).process()
      self.docArr = Normalize(content).process()

  def ExtractTokensFromDoc(self, V, d):
    overlap = []
    for t in d:
      if(t in V): overlap.append(t)

    return overlap
  # ExtractTokensFromDoc({'1','2','3'}, ['2','4'])

  def getHighestScoreClass(self, score):  # score: { 1: 0.123, 2: 0.456, 3: 0.001, ...}
    result = -1  # -1 is an intializer
    firstTime = True

    for c in score.keys():
      if(firstTime):
        maxScore = score[c]
        result = c
        firstTime = False
        continue

      if(score[c] > maxScore):
        maxScore = score[c]
        result = c

    return result
  # getHighestScoreClass({ 1: 0.123, 2: 0.456, 3: 0.001})

  def getClass(self):
    W = self.ExtractTokensFromDoc(SingleDoc.V, self.docArr)
    score = {}

    for c in SingleDoc.C:
      score[c] = log(SingleDoc.prior[c])
      for t in W:
          score[c] += log(SingleDoc.condprob[t][c])
    # print(score)
    return self.getHighestScoreClass(score)

if __name__ == "__main__":
  def gatherAllTrainingDocId(CLASSIFIED_DOCS_PATH = 'training.txt'):
    res = []
    with open(CLASSIFIED_DOCS_PATH, 'r') as f:
      for line in f:
        arr = line.rstrip().split(' ')
        res += arr[1:]
    return res

  with open('output.txt', 'w') as f:
    allDocIds = gatherAllTrainingDocId()

    f.write("{0:<8s}".format('doc_id') + 'class_id' + '\n')
    for i in tqdm(range(1095), desc='Generate [output.txt]'):
      docId = str(i + 1)

      if(docId in allDocIds): continue

      classId = str(SingleDoc(i + 1).getClass())
      f.write("{0:<8s}".format(docId) + classId)
      if(docId != '1095'): f.write('\n')

  print('\n================================ Save outputs in \'./output.txt\' ================================\n')
