predsFile = open('validation.out.testout', 'r')
goldsFile = open('cpbdev.txt', 'r')

preds = [line.split() for line in predsFile.readlines()]
golds = [line.split() for line in goldsFile.readlines()]


outputFile = open('cpbdev.out', 'w')
for predsline, goldsline in zip(preds, golds):
	outline = []
	lastname = ''
	preflag = ''
	for pred, gold in zip(predsline, goldsline):
		newPred = pred
		flag, name = pred[:pred.find('-')], pred[pred.find('-')+1:]
		if flag == 'B':
			lastname = name
		elif flag == 'I' or flag == 'E':
			if name != lastname:
				print(predsline)
				if preflag == '' or preflag == 'O' or preflag == 'S' or preflag == 'E' or preflag == 'rel':
					newPred = 'B-' + name
					lastname = name
				else:
					newPred = flag + '-' + lastname
		preflag = flag
		outline.append(gold + '/' + newPred)
	outputFile.write(' '.join(outline) + '\n')