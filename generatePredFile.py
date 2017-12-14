predsFile = open('validation.out.testout', 'r')
goldsFile = open('cpbdev.txt', 'r')

preds = [line.split() for line in predsFile.readlines()]
golds = [line.split() for line in goldsFile.readlines()]


outputFile = open('cpbdev.out', 'w')
for predsline, goldsline in zip(preds, golds):
	outline = []
	newPreds = []
	lastname = ''
	preflag = ''
	modify = False
	for pred, gold in zip(predsline, goldsline):
		newPred = pred
		flag, name = pred[:pred.find('-')], pred[pred.find('-')+1:]
		if flag == 'B':
			lastname = name
		elif flag == 'I' or flag == 'E':
			if name != lastname:
				newPred = 'B-' + name
				lastname = name
				modify = True
		preflag = flag
		newPreds.append(newPred)
		outline.append(gold + '/' + newPred)
	if modify:
		print('--------------------')
		print(predsline)
		print(newPreds)
		print('--------------------')
	outputFile.write(' '.join(outline) + '\n')