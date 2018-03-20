# hw1
1 a) From terminal: python3 decision_tree_hw1_final.py - will run the main method which extracts features, constructs a tree and returns error rates. You can call the train function in that method a limiting depth and see the resulting output.

1 b) Could have extracted the following features:
  both last and first start with same letter
  last name begins with vowel
  does length of first = length of last
  does last letter of last = last letter of first

1 c) Error rate on training = 6.29%

1 d) Error rate on test = 6.3%

1 e)Max depth = 6

2 a) When running the above, will also cross-validation accuracy for each depth.

2 b) Based on the results of cross-validation, I select a depth of 4 to limit my tree. Results on the test set improved to 5.4%.

2 c) Limiting is a good idea to avoid over-fitting which may lead to poorer results.

