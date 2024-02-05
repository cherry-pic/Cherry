import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# exp       = ["0" , "128", "256", "512", "1024", "summarized", "all"]
# lexRank_f = [None, 0.789, 0.801, 0.788, 0.800, 0.781]   #f-1 results
# GPT3_f =    [None, 0.838, 0.875, 0.880, 0.860, None ]
# longf_b_f = [None, None,  0.836, 0.846, 0.819, None ]
# bert_b_f =  [0.617,0.792, 0.811, 0.840, None,  None ]
#
# lexRank_a = [None, 0.801, 0.810, 0.797, 0.812, 0.789]   # accuracy results
# GPT3_a =    [None, 0.838, 0.875, 0.880, 0.809, None ]
# longf_b_a = [None, None,  0.832, 0.843, 0.813, None ]
# bert_b_a =  [0.619,0.789, 0.826, 0.836, None,  None ]


lexRank_f = [0.789, 0.801, 0.788, 0.800, 0.795, 0.781]   #f-1 results
GPT3_f =    [0.838, 0.875, 0.880, 0.860, 0.842]
longf_b_f = [ 0.836, 0.846, 0.819,0.0]
bert_b_f =  [0.617,0.792, 0.811, 0.840, 0.863]

lexRank_a = [0.801, 0.810, 0.797, 0.812, 0.809, 0.789]   # accuracy results
GPT3_a =    [0.838, 0.875, 0.880, 0.809, 0.779]
longf_b_a = [0.832, 0.843, 0.813,0.0]
bert_b_a =  [0.619,0.789, 0.826, 0.836, 0.840]


lexRank_y = ["128","256","512","1024", "Summary", "Full"]
GPT3_y = ["128","256","512","1024", "Summary"]  # for models with "zero context" experiment
longf_y = ["256","512","1024", "Summary"]
bert_y = ["0","128","256","512","Summary"]

# using subplot function and creating plot one
plt.subplot(1, 2, 1)  # row 1, column 2, count 1
plt.plot(bert_y,bert_b_f, marker = "s")   # plotting
plt.plot(longf_y,longf_b_f,marker = "+")
plt.plot(GPT3_y,GPT3_f,marker = "o")
plt.plot(lexRank_y,lexRank_f,marker = "x")
plt.ylim((0.5, 1))  # range of ticks on y-axis
plt.legend(["BERT-B", "Longformer-B", "GPT 3.5","LexRank"],bbox_to_anchor=(1.05, 1.0),loc='lower right')
#plt.title('FIRST PLOT')
plt.xlabel('Context length (tokens)')
plt.ylabel('F-1')

# using subplot function and creating plot two
# row 1, column 2, count 2
plt.subplot(1, 2, 2)

# g is for green color
plt.plot(bert_y,bert_b_a,marker = "s")   # plotting
plt.plot(longf_y,longf_b_a,marker = "+")
plt.plot(GPT3_y,GPT3_a,marker = "o")
plt.plot(lexRank_y,lexRank_a,marker = "x")
plt.ylim((0.5, 1))  # range of ticks on y-axis
#plt.title('SECOND PLOT')
plt.xlabel('Context length (tokens)')
plt.ylabel('Accuracy')
#plt.legend(["BERT-B", "Longformer-B", "GPT 3.5","LexRank"],bbox_to_anchor=(1.05, 1.0),loc='lower center')

# space between the plots
plt.tight_layout()

# show plot
plt.show()








# fig, (ax1, ax2) = plt.subplot(1, 2, 1)
# ax1.set(xlabel="Context length (tokens)" , ylabel="F-1")
# ax2.set(xlabel="Context length (tokens)" , ylabel="Accuracy")
#
# plt.ylim((0.5, 1))  # range of ticks on y-axis
#
# plt.plot(bert_y,bert_b_f, marker = "s")   # plotting
# plt.plot(longf_y,longf_b_f,marker = "+")
# plt.plot(GPT3_y,GPT3_f,marker = "o")
# plt.plot(lexRank_y,lexRank_f,marker = "x")
#
# fig, (ax1, ax2) = plt.subplot(1, 2, 2)
# plt.plot(bert_y,bert_b_a, marker = "s")   # plotting
# plt.plot(longf_y,longf_b_a,marker = "+")
# plt.plot(GPT3_y,GPT3_a,marker = "o")
# plt.plot(lexRank_y,lexRank_a,marker = "x")
#
#
# plt.legend(["BERT-B", "Longformer-B", "GPT 3.5","LexRank"])  # the marker-name mapping (order should match the order of plotting sentences above)
# plt.show()



