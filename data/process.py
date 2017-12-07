import json
import collections

def splitFile(filename):
	f = open(filename)
	lines = f.readlines()
	retLines = []
	for line in lines:
		tokens = line.split(' ')
		words = []
		poss = []
		srs = []
		for wps in tokens:
			if wps == '\n' or wps == '':
				continue
			word, pos, sr = wps.split('/')
			words.append(word)
			poss.append(pos)
			srs.append(sr)
		retLines.append({'words': words, 'poss': poss, 'srs': srs})
	return retLines

def makeDict(data, key, saveFile):
	tokens = []
	for line in data:
		tokens.extend(line[key])
	counter = collections.Counter(tokens)
	count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
	count_pairs.insert(0, [('<PAD>', 0)])
	f = open(saveFile, 'w')
	for i in range(len(count_pairs)):
		f.write(str(count_pairs[i][0]) + '\t' + str(i) + '\n')
	f.close()

def generateInput(data, saveFile):
	f = open(saveFile, 'w')
	for line in data:
		words = line['words']
		srs = line['srs']
		for i in range(len(words)):
			f.write(words[i] + '\t' + srs[i] + '\n')
		f.write('\n')
	f.close()

data = splitFile('cpbtrain.txt')
#makeDict(data, 'words', 'word2id')
#makeDict(data, 'srs', 'label2id')
generateInput(data, 'train.in')
data = splitFile('cpbdev.txt')
generateInput(data, 'validation.in')