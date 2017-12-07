import json

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

data = splitFile('cpbdev.txt')
file = open('dev.json', 'w')
for line in data:
	file.write(json.dumps(line) + '\n')