import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

font = {'family' : 'normal',
        #'weight': 'bold',
        'size'   : 15}

matplotlib.rc('font', **font)


lexRank_f = [0.798, 0.802, 0.802, 0.802, 0.802]   #f-1 results
GPT3_f =    [0.839, 0.846, 0.844, 0.845, 0.845]
longf_b_f = [0.869, 0.872, 0.877, 0.877, 0.877]
bert_b_f =  [0.880, 0.879, 0.872, 0.872, 0.872]
GPT0_f   =  [0.845, 0.855, 0.852, 0.854, 0.854]   # 0 -shot GPT 3.5

lexRank_a = [0.813, 0.819, 0.819, 0.819, 0.819]    # accuracy results
GPT3_a =    [0.777, 0.787, 0.783, 0.785, 0.785]
longf_b_a = [0.840, 0.836, 0.850, 0.850, 0.850]
bert_b_a =  [0.850, 0.848, 0.842, 0.842, 0.842]
GPT0_a   =  [0.805, 0.815, 0.813, 0.815, 0.815]

x = [100,200,300,400,500]




# using subplot function and creating plot one
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(x,bert_b_f, marker = "s" , markersize=10, markerfacecolor='none')   # plotting
ax1.plot(x,longf_b_f,marker = "+", markersize=10)
ax1.plot(x,GPT3_f,marker = "o", markersize=10, markerfacecolor='none')
ax1.plot(x,GPT0_f,marker = "v", markersize=10, markerfacecolor='none')
ax1.plot(x,lexRank_f,marker = "x", markersize=10)
ax1.set_ylim([0.76, 0.9])  # range of ticks on y-axis
ax1.set_yticks(ax1.get_yticks()[::1])
#ax1.xlabel('Context length (tokens)')
#ax1.ylabel('F-1')
ax1.set_ylabel('F-1')
ax1.set_xlabel('Context length')

ax2.plot(x,bert_b_a,marker = "s", markersize=10, markerfacecolor='none')   # plotting
ax2.plot(x,longf_b_a,marker = "+", markersize=10)
ax2.plot(x,GPT3_a,marker = "o", markersize=10, markerfacecolor='none')
ax2.plot(x,GPT0_a,marker = "v", markersize=10, markerfacecolor='none')
ax2.plot(x,lexRank_a,marker = "x", markersize=10)
ax2.set_ylim([0.76, 0.9])  # range of ticks on y-axis
ax2.set_yticks(ax1.get_yticks()[::1])
#ax2.xlabel('Context length (tokens)')
#ax2.ylabel('Accuracy')
ax2.set_ylabel('Accuracy')
ax2.set_xlabel('Context length')

fig.legend(["BERT-base", "Longformer-base", "GPT 3.5-10 Shots", "GPT 3.5-0 Shots","LexRank" ],bbox_to_anchor=(0.5, 1.025),loc="upper center", ncol=3, frameon=False)
fig.tight_layout(pad=4.0)
# Adjusting the sub-plots
plt.subplots_adjust(right=0.9,  wspace=0.3)

#plt.legend(,bbox_to_anchor=(0.5, 1.0),loc='lower center',ncol=3,columnspacing=1.0,)

#plt.tight_layout()
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



