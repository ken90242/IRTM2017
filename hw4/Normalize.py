import re
import string
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
stopWords = set(stopwords.words('english'))

class Normalize:
  def __init__(self, rawContent):
    self.rawContent = rawContent
    self.tempArr = []
    self.result = []

  def Tokenize(self):
    splitRule = '[\\s|.|,]'
    self.tempArr = re.split(splitRule, self.rawContent)

  # Step 3 - lower case everything
  def LowerCase(self):
    self.tempArr = [str.lower(x) for x in self.tempArr]

  # Step 4 - doing stem jobs
  def Stem(self):
    self.tempArr = [PorterStemmer().stem(x) for x in self.tempArr]

  # Step 5 - get rid of stop words
  def StopWordRemove(self):
    # Downloaded from ('http://ir.dcs.gla.ac.uk/resources/linguistic_utils/stop_words')
    resultArr = []
    for string in self.tempArr:      
      if (string not in stopWords):
        resultArr.append(string)
    self.tempArr = resultArr

  def SpecailCharRemove(self):
    resultArr = []
    for chars in self.tempArr:
      newChars = chars
      for specialChar in string.punctuation:
        newChars = newChars.replace(specialChar, '')
      newChars = re.sub('\d+', '', newChars)
      resultArr.append(newChars)
    resultArr = [x for x in resultArr if x is not '']
    self.tempArr = resultArr

  def process(self):
    self.Tokenize()
    self.LowerCase()
    self.Stem()
    self.SpecailCharRemove()
    self.StopWordRemove()
    self.result = self.tempArr
    return self.result

if __name__ == "__main__":
  Normalize('asd asoiAdja cars car is be are was aaa B bCas v d').process()
