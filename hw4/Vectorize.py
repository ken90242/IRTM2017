import os
import collections
import math
import pickle as pkl
import numpy as np
from tqdm import tqdm
from Normalize import Normalize

class Vectorize:
  def __init__(self, N):
    self.N = N # 1095
    self.df_dict = collections.OrderedDict()
    self.tf_dicts = [None] * (N + 1)
    self.tf_idf_dicts = [None] * (N + 1)
    self.CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))

  def GetFile(self, path):
  	# Downloaded from ('https://ceiba.ntu.edu.tw/course/35d27d/content/28.txt')
    return open(path, 'r').read()

  def TFDFSave(self, arr, tf_dict):
    df_recorded = []
    for term in arr:
      if term in tf_dict:
        tf_dict[term] = tf_dict[term] + 1
      else:
        tf_dict[term] = 1

      if term not in df_recorded:
        if term in self.df_dict:
          self.df_dict[term] = self.df_dict[term] + 1
        else:
          self.df_dict[term] = 1
        df_recorded.append(term)

  def TransferTermToIdx(self, dict, term_idx):
    res_dict = collections.OrderedDict()
    for term in dict:
      idx = term_idx[term]
      res_dict[idx] = dict[term]

    return collections.OrderedDict(sorted(res_dict.items()))

  # Step 6 - save file
  def TFIDFGet(self, tf):
    dict = collections.OrderedDict()
    for term in tf:
      dict[term] = tf[term] * math.log10(self.N / self.df_dict[term])

    return dict

  def TermToIdxMake(self):
    term_idx = collections.OrderedDict()
    idx = 1
    for term in sorted(self.df_dict):
      term_idx[term] = idx
      idx += 1

    return term_idx

  # X,Y is id
  def cosine(self, v1, v2):
    return np.sum(v1 * v2) # element-wise


  def Vectorize(self, dict):
    res = np.zeros(len(self.df_dict))
    total_square = 0
    for k, v in dict.items():
      res[k - 1] = v
      total_square += v * v

    for idx, val in enumerate(res):
      res[idx] = val/math.sqrt(total_square)

    return res

  def process(self):
    for i in tqdm(range(1, self.N + 1), desc='[1/5] Document collection & regularization'):
      self.tf_dicts[i] = collections.OrderedDict()
      fileContent = self.GetFile('./IRTM/' + str(i) + '.txt')
      self.TFDFSave(Normalize(fileContent).process(), self.tf_dicts[i])

    term_idx = self.TermToIdxMake()

    for i in range(1, self.N + 1):
      tmp = self.TFIDFGet(self.tf_dicts[i])
      self.tf_idf_dicts[i] = self.TransferTermToIdx(tmp, term_idx)

    vectorized_dicts = {}
    for docId in tqdm(range(1, self.N + 1), desc='[2/5] Establish tf-idf dictionary'):
      vectorized_dicts[docId] = self.Vectorize(self.tf_idf_dicts[docId])

    d_sim = {}
    for docId, vectors in enumerate(tqdm(self.tf_idf_dicts, desc='[3/5] Establish cosine_similarity dictionary')):
      if(docId == 0): continue
      d_sim[docId] = {}
      for target in range(1, self.N + 1):
        d_sim[docId][target] = self.cosine(vectorized_dicts[docId], vectorized_dicts[target])

    return d_sim
