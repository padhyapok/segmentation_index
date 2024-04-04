Code to order segments along the notochord. Uses manual drawn masks to extract tissue.
Uses pre-processing steps of blurring and background subtraction to detect clusters of expression. 
Then ranks them using size to call the order in which they form. 

Written in Python and uses scipy, scikit-image and opencv.

Uses folder names to collect files from which needs to be updated. Sample data added to the folder. 

Output as text file with an order index.
