#!C:/Python27/python.exe
# Python 2.7.3
# -*- coding: latin-1 -*-

import re
import os
import collections
import time
import cgi

# This is the map where dictionary terms will be stored as keys and value will be posting list with position in the file
dictionary = {}
# This is the map of docId to input file name
docIdMap = {}

print "Content-Type: text/html\r\n\r\n"

class index:
    def __init__(self, path):
        self.path = path
        pass

    def buildIndex(self):

        docId = 1
        fileList = [f for f in os.listdir(self.path) if os.path.isfile(os.path.join(self.path, f))]
        for eachFile in fileList:
            position = 1
            # docName = "Doc_Id_" + str(docId)
            # docName =  str(docId)
            docIdMap[docId] = eachFile
            lines = [line.rstrip('\n') for line in open(self.path + "/" + eachFile)]

            for eachLine in lines:
                wordList = re.split('\W+', eachLine)

                while '' in wordList:
                    wordList.remove('')

                for word in wordList:
                    if (word.lower() in dictionary):
                        postingList = dictionary[word.lower()]
                        if (docId in postingList):
                            postingList[docId].append(position)
                            position = position + 1
                        else:
                            postingList[docId] = [position]
                            position = position + 1
                    else:
                        dictionary[word.lower()] = {docId: [position]}
                        position = position + 1
            docId = docId + 1

    def and_query(self, query_terms):
        print ("<html><head></head><body>")
        if len(query_terms) == 1:
            resultList = self.getPostingList(query_terms[0])
            if not resultList:
                print ""
                printString = "<h2>Result for the Query : <h2 style='color:blue'>" + query_terms[0] + "</h2></h2>"
                print printString
                print "0 documents returned as there is no match"
                return

            else:
                print ""
                printString = "<h2>Result for the Query : <h2 style='color:blue'>" + query_terms[0] + "</h2></h2>"
                print printString
                print " <h2 style='color:red'>Total documents retrieved : " + str(len(resultList))+"</h2>"
                for items in resultList:
                    print "<h3>"
                    print docIdMap[items]
                    print "</h3>"

        else:
            resultList = []
            for i in range(1, len(query_terms)):
                if (len(resultList) == 0):
                    resultList = self.mergePostingList(self.getPostingList(query_terms[0]),
                                                       self.getPostingList(query_terms[i]))
                else:
                    resultList = self.mergePostingList(resultList, self.getPostingList(query_terms[i]))
            print ""
            printString = "<h2>Result for the Query(AND query) :<h2 style='color:blue'>"
            i = 1
            for keys in query_terms:
                if (i == len(query_terms)):
                    printString += " " + str(keys)
                else:
                    printString += " " + str(keys) + " AND"
                    i = i + 1
            printString += "</h2></h2>"
                    
            print printString
            print "<h2 style='color:red'>Total documents retrieved : " + str(len(resultList))+"</h2>"
            for items in resultList:
                print ("<h3>")
                print(docIdMap[items])
                print ("</h3>")
            print "</body></html>"

    def getPostingList(self, term):
        if (term in dictionary):
            postingList = dictionary[term]
            keysList = []
            for keys in postingList:
                keysList.append(keys)
            keysList.sort()
            # print keysList
            return keysList
        else:
            return None

    def mergePostingList(self, list1, list2):

        mergeResult = list(set(list1) & set(list2))
        mergeResult.sort()
        return mergeResult

##    def print_dict(self):
##        # function to print the terms and posting list in the index
##        fileobj = open("invertedIndex.txt", 'w')
##        for key in dictionary:
##            print key + " --> " + str(dictionary[key])
##            fileobj.write(key + " --> " + str(dictionary[key]))
##            fileobj.write("\n")
##        fileobj.close()
##
##    def print_doc_list(self):
##        for key in docIdMap:
##            print "Doc ID: " + str(key) + " ==> " + str(docIdMap[key])

form = cgi.FieldStorage()
query = form.getvalue('query')

def main():
    #docCollectionPath = str(doc)
    #queryFile = str(query)

    with open("query", "w") as text_file:
        text_file.write("%s" % query)
    
    indexObject = index('doc')
    indexObject.buildIndex()

    print ""
    #print "Inverted Index :"
    #indexObject.print_dict()

    print ""
    #print "Document List :"
    #indexObject.print_doc_list()
    print ""

    QueryLines = [line.rstrip('\n') for line in open('query')]
    for eachLine in QueryLines:
        wordList = re.split('\W+', eachLine)

        while '' in wordList:
            wordList.remove('')

        wordsInLowerCase = []
        for word in wordList:
            wordsInLowerCase.append(word.lower())
        indexObject.and_query(wordsInLowerCase)

if __name__ == '__main__':
    main()
