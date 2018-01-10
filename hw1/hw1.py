import re
import os
import string
# Downloaded from ('https://tartarus.org/martin/PorterStemmer/')
from nltk.stem import PorterStemmer

# Step 1 - get document from url
def GetFile():
	# Downloaded from ('https://ceiba.ntu.edu.tw/course/35d27d/content/28.txt')
  return open('./rawContent.txt', 'r').read()

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

# Step 5 - get rid of stop words
def StopWordRemove(arr):
  # Downloaded from ('http://ir.dcs.gla.ac.uk/resources/linguistic_utils/stop_words')
  stopWords = open('./stopWordList.txt', 'r').read().split('\n')
  resultArr = []
  for string in arr:
    if (string not in stopWords):
      resultArr.append(string)
  return resultArr

def SpecailCharRemove(arr):
  resultArr = []
  for chars in arr:
    newChars = chars
    for specialChar in string.punctuation:
      newChars = newChars.replace(specialChar, "")
    resultArr.append(newChars)
  return resultArr

# Step 6 - save file
def ContentSave(arr):
  CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
  target = open("result.txt", "w")
  combinedContent = '\n'.join(arr)
  target.write(combinedContent)

  print ('Save result at [ ' + CURRENT_PATH + '/result.txt ]')

# Execute assignment's requirement step by step
fileContent = GetFile()
TokenizedContent = Tokenize(fileContent)
LowerCasedContent = LowerCase(TokenizedContent)
StemmedContent = Stem(LowerCasedContent)
StopWordRemovedContent = StopWordRemove(StemmedContent)
SpecailCharRemovedContent = SpecailCharRemove(StopWordRemovedContent)
ContentSave(SpecailCharRemovedContent)
