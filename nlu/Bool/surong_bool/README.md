# booleanRetrieval
                                              
===========================================================================================================
1) Unzip given file
2) Make sure you have python installed on your pc. I have used 2.7.3 version
3) Open terminal
4) Excute below commands
    a) python index.py
    b) It will ask you to enter directory path containing input text files. Paste the directory path and press enter
        Example - "doc"
    c) It will ask you to enter path of the query file (file, where all the queries are written). Refer "query" file
        Example - "query"
    d) It will build index, print time taken to build index, dictionary and doc id to filename map
    e) It will read every line of queries file, treat each line as a new query and perform search on them
    f) prints search result along with time taken.
    g) Have a look at attached "invertedIndex.txt" file to view created inverted index.


                                            Algorithm
=============================================================================================================
A) Build Index :
    1) This method gets invoked as soon as user inputs dictionary path
	2) Iterates through all the files in the dictionary
	3) For each file, read every line and stores it in lines list.
	4) Then iterates through lines list and split each line using '\W+' de limiter
	5) Removes all the empty words and iterates through wordsList
	6) Convert each word to lower case and search for word in dictionary keys.
	7) If word is present in dictionary, search for document Id.
	8) If document id is same as current doc id, append the position of the word to posting list, else create a new map with
		docId as key and position as value.
	9) If word is not present in dictionary, create a new entry in map with word as key, it's docId and position as value
	10) Example map ->
			"word" : {docId : [position_List], docId : [positionList]}
	11) Each file is given a unique doc id
	12) position variable is used to calculate position of all words and is initialized to 1 everytime a new file is opened to read

B) Merge :
    1) and_query method takes list of query terms as input
	2) If the length of query_terms is equal to 1, It will get the posting list for the only term. If no result is found,
	   appropriate message will be printed. Else, print the respective file name by traversing docIdToFileName map
	   for all the posting lists.
	3) If there are more than 1 item in query_terms, it will get posting list for first and second term. Then calls mergePostingList
	    ("This function takes 2 lists as input parameters and returns intersection of both in a sorted form") by passing posting lists of term0 and term1.
	    The merge result will be used to get intersection of posting list of term3.
	4) This will be repeated for all the subsequent query terms.
	5) Result of step 2 will have merged list of all the query terms
	6) Uses the final merged list to traverse through DocIdToFileName map and prints respective file names


	                                            Performance
=============================================================================================================
    1) Avg time taken to build index for given set of text file --> ~1 to ~1.5 seconds
    2) Time taken by query "with AND without AND yemen" --> ~0.000185 seconds
    3) Time taken by query "with AND without AND yemen AND yemeni" --> ~0.0001540 seconds
