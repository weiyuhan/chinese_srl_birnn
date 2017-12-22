#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os
import helper

def generate(predsPath, goldsPath, outPath):
	predsFile = open(predsPath, 'r')
	goldsFile = open(goldsPath, 'r')

	preds = [line.split() for line in predsFile.readlines()]
	golds = [line.split() for line in goldsFile.readlines()]

	outputFile = open(outPath, 'w')
	modify_count = 0
	outPreds = []
	for predsline, goldsline in zip(preds, golds):
		outline = []
		for pred, gold in zip(predsline, goldsline):
			outline.append(gold + '/' + pred)
		outPreds.append(outline)
		
	helper.regularPred(outPreds)

	for outline in outPreds:
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