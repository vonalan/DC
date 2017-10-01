# -*- coding: utf-8 -*-

def writeLog(file, string): 
    wf = open(file, 'a')
    wf.write(string)
    wf.write('\n')
    wf.close()
