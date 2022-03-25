import time
from tqdm import tqdm
import torch
import copy

def b_train(num_epochs,train_loader,test_loader,device,net,loss,optimizer,train_writer,test_writer):
    """迭代次数，训练数据，测试数据，设备， 网络， 损失， 优化器， 训练日志， 测试日志"""
    best_test_acc = 0
    best_net_params = copy.deepcopy(net.state_dict())
    for epoch in range(num_epochs):
#训练损失和，训练精确度，样本数，batcht数量， 记录当前训练次数开始时间
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        #for X, y in train_loader:
        for X, y in tqdm(train_loader):#和上一句功能一致，但是会显示进度条
            #print(X.shape)
            X = X.to(device)#数据复制到设备
            y = y.to(device)
            y_hat = net(X)#预测输出
            l = loss(y_hat, y)#损失
            optimizer.zero_grad()#梯度置零
            l.backward()#反向传播求导
            optimizer.step()#优化
            train_l_sum += l.cpu().item() * y.shape[0]#loss复制到cpu上，并由train_l_sum记录
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()#由train_acc_sum记录预测正确个数
            n += y.shape[0]#记录样本个数,可以用 len(test_loader.dataset)
            #batch_count += 1#batch次数加1
        trian_acc = train_acc_sum / n
        train_l = train_l_sum / n

        #计算当前的测试样本精确度
        net.eval()#切换到评价模式
        acc_sum, test_l_sum, n = 0.0, 0.0, 0
        with torch.no_grad():#不累计grad，释放显存
            for X, y in test_loader:
                y_hat = net(X.to(device))
                y = y.to(device)
                l = loss(y_hat, y)#损失
                test_l_sum += l.cpu().item() * y.shape[0]#loss复制到cpu上，并由train_l_sum记录
                acc_sum += (y_hat.argmax(dim=1) == y).float().sum().cpu().item()
                n += y.shape[0]
            test_acc = acc_sum / n
            test_l = test_l_sum / n
        net.train() # 改回训练模式

        #打印训练一轮后效果
        train_writer.add_scalar('loss', train_l, epoch + 1)
        test_writer.add_scalar('loss', test_l, epoch + 1)
        train_writer.add_scalar('acc', trian_acc, epoch + 1)
        test_writer.add_scalar('acc', test_acc, epoch + 1)
        print('epoch %d, train loss %.5f, train acc %.5f, test loss %.5f, test acc %.5f, time %.2f sec'
            % (epoch + 1, train_l, trian_acc, test_l, test_acc, time.time() - start))
        if best_test_acc < test_acc:
            best_test_acc = test_acc
            best_net_params = copy.deepcopy(net.state_dict())
            torch.save(best_net_params,"temp_best_models.pt")
    print("Best Test Acc: {}".format(best_test_acc))#打印最好结果
    return best_net_params