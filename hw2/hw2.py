import re
import os
import string
import collections
import math
# Downloaded from ('https://tartarus.org/martin/PorterStemmer/')
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
stopWords = set(stopwords.words('english'))
from tqdm import tqdm


N = 1095
df_dict = collections.OrderedDict()
tf_dicts = [None] * (N + 1)
tf_idf_dicts = [None] * (N + 1)

# Step 1 - get document from url
def GetFile(path):
	# Downloaded from ('https://ceiba.ntu.edu.tw/course/35d27d/content/28.txt')
  return open(path, 'r').read()

# Step 2 - tokenize
def Tokenize(multiStrings):
  def tokenizeFilter(singleString):
    return singleString if singleString != '' else None

  splitRule = '[\s|.|,]'
  splitedArr = re.split(splitRule, multiStrings)
  resultArr = filter(tokenizeFilter, splitedArr)
  return resultArr

# Step 3 - lower case everything
def LowerCase(arr):
  resultArr = map(str.lower, arr)
  return resultArr

# Step 4 - doing stem jobs
def Stem(arr):
  stemFunc = PorterStemmer().stem
  def stemMapper(string):
    return stemFunc(string)

  resultArr = map(stemMapper, arr)
  return resultArr

def StopWordRemove(arr):
  resultArr = []
  for string in arr:
    if (string not in stopWords):
      resultArr.append(string)
  return resultArr

def SpecailCharRemove(arr):
  resultArr = []
  for chars in arr:
    newChars = chars
    # print(newChars)
    for idx, specialChar in enumerate(string.punctuation):
      if(specialChar=='-'):
        if(idx!=0 and idx!=len(string.punctuation)):
          newChars = newChars.replace(specialChar, "")
      else:
        newChars = newChars.replace(specialChar, "")
    # print (digit_pattern.match('asd123qweqw'))
    # if newChars != '' and (not digit_pattern.match(newChars)):
    # print(newChars)
    if(newChars != ''):
      resultArr.append(newChars)
  return resultArr

def TFDFSave(arr, tf_dict, df_dict):
  df_recorded = []
  for term in arr:
    if term in tf_dict:
      tf_dict[term] = tf_dict[term] + 1
    else:
      tf_dict[term] = 1

    if term not in df_recorded:
      if term in df_dict:
        df_dict[term] = df_dict[term] + 1
      else:
        df_dict[term] = 1
      df_recorded.append(term)

def TransferTermToIdx(dict):
  res_dict = collections.OrderedDict()
  for term in dict:
    idx = term_idx[term]
    res_dict[idx] = dict[term]

  return collections.OrderedDict(sorted(res_dict.items()))

# Step 6 - save file
def TFIDFGet(tf, df):
  dict = collections.OrderedDict()
  for term in tf:
    dict[term] = tf[term] * math.log10(N / df_dict[term])

  return dict

def TermToIdxMake(df_dict):
  term_idx = collections.OrderedDict()
  idx = 1
  for term in sorted(df_dict):
    term_idx[term] = idx
    idx += 1

  return term_idx

# X,Y is id
def cosine(DocX, DocY):
  # list
  v1 = Vectorize(tf_idf_dicts[DocX])
  v2 = Vectorize(tf_idf_dicts[DocY])
  res = 0

  for idx in range(len(v1)):
    res += v1[idx] * v2[idx]
  return res


def Vectorize(dict):
  res = []
  total_square = 0
  for i in range(len(df_dict)):
    res.append(0.0)
  for k, v in dict.items():
    res[k - 1] = v
    total_square += v * v

  for idx, val in enumerate(res):
    res[idx] = val/math.sqrt(total_square)

  return res


for i in tqdm(range(1, N + 1), desc='(1/4) Document collection & regularization'):
  tf_dicts[i] = collections.OrderedDict()
  fileContent = GetFile('./IRTM/' + str(i) + '.txt')
  TokenizedContent = Tokenize(fileContent)
  LowerCasedContent = LowerCase(TokenizedContent)
  StemmedContent = Stem(LowerCasedContent)
  StopWordRemovedContent = StopWordRemove(StemmedContent)
  SpecailCharRemovedContent = SpecailCharRemove(StopWordRemovedContent)
  TFDFSave(SpecailCharRemovedContent, tf_dicts[i], df_dict)

term_idx = TermToIdxMake(df_dict)

for i in tqdm(range(1, N + 1), desc='(2/4) Establish tf-idf dictionary'):
  tmp = TFIDFGet(tf_dicts[i], df_dict)
  tf_idf_dicts[i] = TransferTermToIdx(tmp)


CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
f = open("dictionary.txt", "w")
f.write("{0:<10s}".format('t_index')+"{0:<16s}".format('term')+"{0:<6s}\n".format('df'))
for i in tqdm(range(1, len(term_idx) + 1), desc='(3/4) Write dictionary.txt(frequency file)'):
  term = list(term_idx.keys())[i - 1]
  if(i == len(term_idx)):
    f.write("{0:<10s}".format(str(i))+"{0:<16s}".format(term)+"{0:<6s}".format(str(df_dict[term])))
  else:
    f.write("{0:<10s}".format(str(i))+"{0:<16s}".format(term)+"{0:<6s}\n".format(str(df_dict[term])))
f.close()

dictionary = tf_idf_dicts[1]
f = open("1.txt", "w")
f.write(str(len(dictionary)) + '\n')
f.write("{0:<10s}".format('t_index') + "{0:<6s}\n".format('tf-idf'))
for i in tqdm(dictionary, desc='(4/4) Write 1.txt(doc1 vector file)'):
  if(i == len(dictionary)):
    f.write("{0:<10s}".format(str(i)) + "{0:<6s}".format(str(dictionary[i])))
  else:
    f.write("{0:<10s}".format(str(i)) + "{0:<6s}\n".format(str(dictionary[i])))
f.close()


print ('total:'+str(len(df_dict)))
print (cosine(1, 2))
