from sklearn.metrics import roc_curve, auc, mean_squared_error, accuracy_score
def check_fit(truth, prob):
    """
    truth: 真实的值 [1,0,1,1,1]
    prob: 预测的值 [0.9,0.7,0.8,0.2,0.3]
    """
    fpr, tpr, _ = roc_curve(truth, prob)     # drop_intermediate:(default=True) 
    roc_auc = auc(fpr, tpr)   # 计算auc值，roc曲线下面的面积 等价于 roc_auc_score(truth,prob)
 
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.1, 1.05])
    plt.ylim([-0.1, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show() 
 
    print('results are RMSE, accuracy, ROC')
    predics = [1 if i>=0.5 else 0 for i in prob]
    print(math.sqrt(mean_squared_error(truth, prob)), accuracy_score(truth, predics), roc_auc)



def calAUC(prob,labels):
    f = list(zip(prob,labels))
    rank = [values2 for values1,values2 in sorted(f,key=lambda x:x[0])]
    rankList = [i+1 for i in range(len(rank)) if rank[i]==1]
    posNum = 0
    negNum = 0
    for i in range(len(labels)):
        if(labels[i]==1):
            posNum+=1
        else:
            negNum+=1
    auc = 0
    auc = (sum(rankList)- (posNum*(posNum+1))/2)/(posNum*negNum)
    print(auc)
    return auc
