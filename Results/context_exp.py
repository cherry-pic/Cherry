import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

font = {'family' : 'normal',
        #'weight': 'bold',
        'size'   : 15}

matplotlib.rc('font', **font)


lexRank_f = [0.789, 0.795, 0.801, 0.788, 0.800, 0.781]   #f-1 results
GPT3_f =    [0.838, 0.842, 0.875, 0.880, 0.860]
longf_b_f = [0.830, 0.865, 0.836, 0.846, 0.819]
bert_b_f =  [0.617, 0.792, 0.863, 0.811, 0.840]
GPT0_f   =  [0.711, 0.824, 0.807, 0.823, 0.852]              # 0 -shot GPT 3.5

lexRank_a = [0.801, 0.809, 0.810, 0.797, 0.812, 0.789]   # accuracy results
GPT3_a =    [0.805, 0.779, 0.834, 0.842, 0.809]
longf_b_a = [0.793, 0.826, 0.832, 0.843, 0.813]
bert_b_a =  [0.619, 0.789, 0.840, 0.826, 0.836]
GPT0_a   =  [0.713, 0.797, 0.789, 0.803, 0.826]              # 0 -shot GPT 3.5

lexRank_y = ["128","Sum","256","512","1024","Full"]
GPT3_y =    ["128","Sum","256","512","1024"]  # 10 shot GPT 3.5
longf_y =   ["128","Sum", "256","512","1024"]
bert_y =    ["0","128","Sum","256","512"]
GPT0_y =    ["128","Sum","256","512","1024"]   # 0 -shot GPT 3.5




# using subplot function and creating plot one
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(bert_y,bert_b_f, marker = "s")   # plotting
ax1.plot(longf_y,longf_b_f,marker = "+")
ax1.plot(GPT3_y,GPT3_f,marker = "o")
ax1.plot(GPT0_y,GPT0_f,marker = "v")
ax1.plot(lexRank_y,lexRank_f,marker = "x")
#ax1.ylim((0.6, 0.9))  # range of ticks on y-axis
#ax1.xlabel('Context length (tokens)')
#ax1.ylabel('F-1')
ax1.set_ylabel('F-1')
ax1.set_xlabel('Context length')

ax2.plot(bert_y,bert_b_a,marker = "s")   # plotting
ax2.plot(longf_y,longf_b_a,marker = "+")
ax2.plot(GPT3_y,GPT3_a,marker = "o")
ax2.plot(GPT0_y,GPT0_a,marker = "v")
ax2.plot(lexRank_y,lexRank_a,marker = "x")
#ax2.ylim((0.6, 0.9))  # range of ticks on y-axis
#ax2.xlabel('Context length (tokens)')
#ax2.ylabel('Accuracy')
ax2.set_ylabel('Accuracy')
ax2.set_xlabel('Context length')

fig.legend(["BERT-B", "Longformer-B", "GPT 3.5-10 Shots", "GPT 3.5-0 Shots","LexRank" ],bbox_to_anchor=(0.5, 1.02),loc="upper center", ncol=3, frameon=False)

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



