predsFile = open('validation.out.testout', 'r')
goldsFile = open('cpbdev.txt', 'r')

preds = [line.split() for line in predsFile.readlines()]
golds = [line.split() for line in goldsFile.readlines()]


outputFile = open('cpbdev.out', 'w')
for predsline, goldsline in zip(preds, golds):
	outline = []
	for pred, gold in zip(predsline, goldsline):
		outline.append(gold + '/' + pred)
	outputFile.write(' '.join(outline) + '\n')