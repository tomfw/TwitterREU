#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import sys 

class Line: 
	def __init__(self, *args, **kwargs):
		self.lineParamDict = {}
		self.normalizeParamDict = {}
		self.concatenateParamDict = {}
		self.lineEXE = os.environ['LINEEXEFILE']
		self.normalizeEXE = os.environ['NORMALIZEEXEFILE']
		self.concateEXE = os.environ['CONCATEXE']

	def initLineParams(self):
		"""
		Samples are in millions
		"""
		self.lineParamDict["train"] = ""
		self.lineParamDict["output"] = ""
		self.lineParamDict["binary"] = 0
		self.lineParamDict["size"] = 128 
		self.lineParamDict["order"] = 1
		self.lineParamDict["negative"] = 5
		self.lineParamDict["samples"] = 1 
		self.lineParamDict["threads"] = 4
		self.lineParamDict["rho"] = 0.025 
		return self.lineParamDict

	def initnormalizeParamDict(self):
		self.normalizeParamDict["input"] = ""
		self.normalizeParamDict["output"] = ""
		self.normalizeParamDict["binary"] = 0
		return self.normalizeParamDict

	def initconcatenateParamDict(self):
		self.concatenateParamDict["input1"] = ""
		self.concatenateParamDict["input2"] = "" 
		self.concatenateParamDict["output"] = ""
		self.concatenateParamDict["-binary"] = 0
		return self.concatenateParamDict

	def buildArgListforLine(self, lPDict):
		args = [self.lineEXE, "-train", lPDict["train"], "-output", lPDict["output"],\
			"-binary",str(lPDict["binary"]),"-size", str(lPDict["size"]),\
			"-order", str(lPDict["order"]), "-negative", str(lPDict["negative"]),\
			"-samples", str(lPDict["samples"]), "-threads", str(lPDict["threads"]),\
				"-rho", str(lPDict["rho"])]
		return args 

	def buildArgListforNormalize(self, nPDict):
		args = [self.normalizeEXE, "-input", self.nPDict['input'],\
			"-output", self.nPDict['output'], "-binary",str(self.nPDict["binary"])]
		return args 

	def buildArgListforConcat(self, cPDict):
		args = [self.concateEXE, "-input1", cPDict["input1"],\
			"-input2", cPDict["input2"], "-output", cPDict["output"],\
			"-binary", str(cPDict["binary"])]
		return args 
