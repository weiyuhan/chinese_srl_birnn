#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os

def generate(predsPath, goldsPath, outPath):
	predsFile = open(predsPath, 'r')
	goldsFile = open(goldsPath, 'r')

	preds = [line.split() for line in predsFile.readlines()]
	golds = [line.split() for line in goldsFile.readlines()]

	outputFile = open(outPath, 'w')
	modify_count = 0
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
			modify_count += 1
			print('--------------------')
			print(predsline)
			print(newPreds)
			print('--------------------')
		outputFile.write(' '.join(outline) + '\n')
	print('total: %5d, modify: %5d' % (len(preds), modify_count))

if __name__ == "__main__":
    if len(sys.argv[1:]) != 3:
        print('the function takes exactly three parameters: predPath, goldPath and outPath')
    else:
        if not os.path.exists(sys.argv[1]):
            print('pred_file not exists!')
        elif not os.path.exists(sys.argv[2]):
            print('gold_file not exists!')
        else:
            generate(sys.argv[1], sys.argv[2], sys.argv[3])