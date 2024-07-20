import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

font = {'family' : 'normal',
        #'weight': 'bold',
        'size'   : 15}

matplotlib.rc('font', **font)


lexRank_f = [0.809, 0.808, 0.809, 0.806, 0.807]   # F-1
GPT3_f =    [0.874, 0.870, 0.868, 0.873, 0.862]
longf_b_f = [0.849, 0.875, 0.858, 0.862, 0.885]
bert_b_f =  [0.870, 0.870, 0.860, 0.871, 0.882]
GPT0_f   =  [0.868, 0.845, 0.832, 0.868, 0.836]

lexRank_a = [0.821, 0.822, 0.821, 0.821, 0.820]    # ACCURACY
GPT3_a =    [0.832, 0.824, 0.822, 0.830, 0.813]
longf_b_a = [0.822, 0.840, 0.818, 0.838, 0.859]
bert_b_a =  [0.846, 0.834, 0.836, 0.844, 0.861]
GPT0_a   =  [0.842, 0.815, 0.805, 0.844, 0.803]   # 0 -shot GPT 3.5

x = [100,200,300,400,500]
# lexRank_y = ["100","200","300","400","500"]
# GPT3_y =    ["128","256","512","1024"]  # 10 shot GPT 3.5
# longf_y =   ["128","256","512","1024"]
# bert_y =    ["0","128","256","512"]
# GPT0_y =    ["128","256","512","1024"]   # 0 -shot GPT 3.5




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



