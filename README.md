# 二手车交易价格预测经验总结

1.数据处理

     1.1神经网络数据处理：
        1）空值与异常值的处理的处理：填充的是众数   
        2）特征之间的简单结合，如相加
        3）提取出日期信息
        4）count_编码与分桶操作
        5）用连续数值特征对类别特征刻画 与 类别与类别特征之间的刻画
        6）对某些类别进行mean-encoding编码 与 target-encoding编码
        7）特征归一化
        8）用pca对数据进行降维    
        
    1.2树模型数据处理
        1）空值与异常值的处理的处理：填充的是众数
        2）处理预测值price长尾分布的问题，取log，使其变成正态分布
        3）提取出日期信息与城市信息
        4）用price连续特征对类别特征进行刻画，选出来很多对特征，然后可以用lgb选出好的分类特征
        5）对v0-v14连续特征进行特征融合，如+，*
        6）筛选特征，可以用lgb
    
    1.3特征处理总结
        1.3.1 类别特征
            1）对类别进行count编码
            2）用连续特征对类别特征进行刻画（包括min，max，std等等编码、以及可以mean-encoding与target-encoding编码）
            3）类别与类别特征进行刻画，如两个类别的共现次数
        1.3.2 连续特征
            1）连续特征进行分桶操作
            2）连续与连续特征、连续与离散特征进行特征融合（+或者*）

2.模型

    2.1NN模型（简单的MLP全连接层，根据epoch进行学习率衰减）
    
    2.2树模型
        1）lgb模型
        2）cat模型

3.模型融合（stack+mix）

    将数模型的预测结果stack起来，然后与NN模型的结果mix起来



其他文件夹：

    建立data文件夹，上传二手车交易数据
    
    建立user_data文件夹，存放中间结果（包括神经网络模型与树模型特征输出等等）



