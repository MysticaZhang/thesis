
# coding: utf-8

# ### 数据分类格式
import pandas as pd
# ### user_id在train和test中有没有重复？
# 读取数据
data = pd.read_csv("tap_fun_train.csv", parse_dates=True)
data_test = pd.read_csv("tap_fun_test.csv", parse_dates=True)

# 提取user_id列，并做合并处理
data_id = pd.DataFrame(data['user_id'],columns=['user_id'])
data_test_id = pd.DataFrame(data_test['user_id'],columns=['user_id'])
pd.merge(data_id, data_test_id, on = 'user_id')

## 训练集data清理及分类
data

from pyecharts import Bar, Overlap

line3 = Line()
line3.add("注册玩家数量", data_day_count.index, data_day_count['register_time_day'], mark_line=["average"], mark_point=["max", "min"])

bar = Bar()
bar.add("45天内付费玩家比例", data_day_count.index, data_day_count['pay_percent'], yaxis_max=0.1, mark_point=["max", "min"])


overlap = Overlap()
# 默认不新增 x y 轴，并且 x y 轴的索引都为 0
overlap.add(line3)
# 新增一个 y 轴，此时 y 轴的数量为 2，第二个 y 轴的索引为 1（索引从 0 开始），所以设置 yaxis_index = 1
# 由于使用的是同一个 x 轴，所以 x 轴部分不用做出改变
overlap.add(bar, yaxis_index=1, is_add_yaxis=True)
overlap.render()

overlap


data_pay_45 = copy.copy(data[data['prediction_pay_price']!=0])
data_pay_45['prediction_pay_price']


from pyecharts import Line, Grid

# 增加两列
data['register_time_month'] = data.register_time.str[:7]
data['register_time_day'] = data.register_time.str[6:10]

# 统计并保存为dataframe
data_month_df = pd.DataFrame(data['register_time_month'].value_counts()).sort_index()
# print(data_month_df)
data_day_df = pd.DataFrame(data['register_time_day'].value_counts()).sort_index()
# print(data_day_df)

from pyecharts import Line, Grid

line1 = Line("玩家数量统计-月")
line1.add("玩家数量", data_month_df.index, data_month_df['register_time_month'], mark_line=["average"], mark_point=["max", "min"])

line2 = Line("玩家数量统计-日",title_top="50%")
line2.add("玩家数量", data_day_df.index, data_day_df['register_time_day'], mark_line=["average"], mark_point=["max", "min"])

grid = Grid(width = 1000, height = 1000)
grid.add(line1, grid_bottom="60%")
grid.add(line2, grid_top="60%")
grid.render()

grid

import copy

data_pay_7 = copy.copy(data[data['pay_price']>0])
print(data_pay_7.shape)   # (41439, 111)
print(data_pay_7.shape[0]/data.shape[0])  # 0.018111395638212645

data_pay_7_day_df = pd.DataFrame(data_pay_7['register_time_day'].value_counts()).sort_index()
# print(data_pay_7_day_df)
data_pay_7_day_df.rename(columns={'register_time_day':'pay_register_time_day'}, inplace = True)
data_day_count = pd.concat([data_pay_7_day_df, data_day_df], axis=1)
# print(data_day_count)
data_day_count['pay_percent'] = data_day_count['pay_register_time_day']/data_day_count['register_time_day']
# print(data_day_count)

from pyecharts import Overlap

line3 = Line()
line3.add("注册玩家数量", data_day_count.index, data_day_count['register_time_day'], mark_line=["average"], mark_point=["max", "min"])

line4 = Line()
line4.add("7天内付费玩家数量", data_day_count.index, data_day_count['pay_register_time_day'], mark_line=["average"], 
          mark_point=["max", "min"], yaxis_max=3000)

overlap = Overlap()
# 默认不新增 x y 轴，并且 x y 轴的索引都为 0
overlap.add(line3)
# 新增一个 y 轴，此时 y 轴的数量为 2，第二个 y 轴的索引为 1（索引从 0 开始），所以设置 yaxis_index = 1
# 由于使用的是同一个 x 轴，所以 x 轴部分不用做出改变
overlap.add(line4, yaxis_index=1, is_add_yaxis=True)
overlap.render()

overlap


from pyecharts import Bar, Overlap

line3 = Line()
line3.add("注册玩家数量", data_day_count.index, data_day_count['register_time_day'], mark_line=["average"], mark_point=["max", "min"])

bar = Bar()
bar.add("7天内付费玩家比例", data_day_count.index, data_day_count['pay_percent'], yaxis_max=0.1)


overlap = Overlap()
# 默认不新增 x y 轴，并且 x y 轴的索引都为 0
overlap.add(line3)
# 新增一个 y 轴，此时 y 轴的数量为 2，第二个 y 轴的索引为 1（索引从 0 开始），所以设置 yaxis_index = 1
# 由于使用的是同一个 x 轴，所以 x 轴部分不用做出改变
overlap.add(bar, yaxis_index=1, is_add_yaxis=True)
overlap.render()

overlap

# #### 会给多少钱
data_pay_45 = copy.copy(data[data['prediction_pay_price']!=0])
print(data_pay_45['prediction_pay_price'].describe())
print('前45天合共付费：',data_pay_45['prediction_pay_price'].sum())

data_pay_7 = copy.copy(data[data['pay_price']!=0])
print(data_pay_7['pay_price'].describe())
print('前7天合共付费：',data_pay_7['pay_price'].sum())  
print('前45天合共付费：',data_pay_7['prediction_pay_price'].sum())

data_nopay_7_pay_45 = copy.copy(data_pay_45[data_pay_45['pay_price']==0])
print(data_nopay_7_pay_45['prediction_pay_price'].describe())
print('前七天没有，后45天有付款的合共付费：',data_nopay_7_pay_45['prediction_pay_price'].sum())


data_pay_7_nopay_45 = copy.copy(data_pay_7[data_pay_7['pay_price']==data_pay_7['prediction_pay_price']])
print(data_pay_7_nopay_45['pay_price'].describe())
print(data_pay_7_nopay_45['pay_count'].describe())
print('前7天合共付费：',data_pay_7_nopay_45['pay_price'].sum())  
print('前7天给钱了，但是后面45天不再给钱的：',data_pay_7_nopay_45.shape[0])

data_pay_7_pay_45 = copy.copy(data_pay_7[data_pay_7['pay_price']<data_pay_7['prediction_pay_price']])
print(data_pay_7_pay_45['pay_price'].describe())
print(data_pay_7_pay_45['pay_count'].describe())
print('前7天合共付费：',data_pay_7_pay_45['pay_price'].sum())  
print('前45天合共付费：',data_pay_7_pay_45['prediction_pay_price'].sum())  
print('前7天给钱了，后面45天继续给钱的：',data_pay_7_pay_45.shape[0])


# #### 在线时长


data['avg_online_minutes'].describe()
data_on=data[data['avg_online_minutes']!=0]
data=data.sort_values(by="avg_online_minutes",ascending= True)
plt.scatter(range(1,2245153),data_on['avg_online_minutes'])
#plt.xlim(-6,6)
plt.plot

plt.yscale('log')
#plt.xscale('log')

data_pay = copy.copy(data[data['pay_price']!=0]) #付费玩家的df
data_non_pay = copy.copy(data[data['pay_price']==0]) #非付费玩家的df

plt.boxplot(x=data_pay['avg_online_minutes'],whis=1.5,meanline=True)
plt.title('Paid Players')
plt.style.use('ggplot')  
plt.show()

plt.boxplot(x=data_non_pay['avg_online_minutes'],whis=1.5,meanline=True)
plt.title('NON-paid players')
plt.ylim(0,10)
plt.style.use('ggplot') 
plt.show()

data['avg_online_minutes'].describe()

data_pay = copy.copy(data[data['pay_price']!=0])
# data_pay.shape
data_pay['avg_online_minutes'].describe()

data_non_pay['avg_online_minutes'].describe()

# #### 单次付费有什么额度的？
data_once = copy.copy(data[data['pay_count']==1])
# data_once.shape  
data_once.groupby("pay_price")["pay_count"].sum()

data_test['register_time_month'] = data_test.register_time.str[:7]
data_test['register_time_day'] = data_test.register_time.str[6:10]

# 统计并保存为dataframe
data_test_month_df = pd.DataFrame(data_test['register_time_month'].value_counts()).sort_index()
# print(data_month_df)
data_test_day_df = pd.DataFrame(data_test['register_time_day'].value_counts()).sort_index()
# print(data_day_df)


#资源类

#付费玩家与非付费玩家df
data_pay = copy.copy(data[data['pay_price']!=0]) #付费玩家的df
data_non_pay = copy.copy(data[data['pay_price']==0]) #非付费玩家的df

sum(data_pay['wood_reduce_value']
    +data_pay['stone_reduce_value']
    +data_pay['ivory_reduce_value']
    +data_pay['meat_reduce_value']
    +data_pay['magic_reduce_value'])/41439
#付费玩家资源总使用量 1097873603508.0
#付费玩家资源人均使用量 26493728.215159632


sum(data_pay['wood_add_value']
    +data_pay['stone_add_value']
    +data_pay['ivory_add_value']
    +data_pay['meat_add_value']
    +data_pay['magic_add_value'])/41439
#付费玩家资源总添加量 1544094355421.0
#付费玩家资源人均添加量 37261863.35145636

sum(data_pay['wood_reduce_value']
    +data_pay['stone_reduce_value']
    +data_pay['ivory_reduce_value']
    +data_pay['meat_reduce_value']
    +data_pay['magic_reduce_value'])/sum(data_pay['wood_add_value']
    +data_pay['stone_add_value']
    +data_pay['ivory_add_value']
    +data_pay['meat_add_value']
    +data_pay['magic_add_value'])
#付费玩家资源使用率0.711014582530912

sum(data_non_pay['wood_reduce_value']
    +data_non_pay['stone_reduce_value']
    +data_non_pay['ivory_reduce_value']
    +data_non_pay['meat_reduce_value']
    +data_non_pay['magic_reduce_value'])/2246568
#非付费玩家资源总使用量1065772239555.0
#非付费玩家资源人均使用量 474400.1693049131

sum(data_non_pay['wood_add_value']
    +data_non_pay['stone_add_value']
    +data_non_pay['ivory_add_value']
    +data_non_pay['meat_add_value']
    +data_non_pay['magic_add_value'])/2246568
#非付费玩家资源总添加量 1626504258310.0
#非付费玩家资源人均添加量 723995.1153537306

sum(data_non_pay['wood_reduce_value']
    +data_non_pay['stone_reduce_value']
    +data_non_pay['ivory_reduce_value']
    +data_non_pay['meat_reduce_value']
    +data_non_pay['magic_reduce_value'])/sum(data_non_pay['wood_add_value']
    +data_non_pay['stone_add_value']
    +data_non_pay['ivory_add_value']
    +data_non_pay['meat_add_value']
    +data_non_pay['magic_add_value'])
#非付费玩家资源使用率0.6552532734604569


#资源分细节
sum(data_non_pay['wood_reduce_value'])/sum(data_non_pay['wood_add_value'])
sum(data_non_pay['stone_reduce_value'])/sum(data_non_pay['stone_add_value'])
sum(data_non_pay['ivory_reduce_value'])/sum(data_non_pay['ivory_add_value'])
sum(data_non_pay['meat_reduce_value'])/sum(data_non_pay['meat_add_value'])
sum(data_non_pay['magic_reduce_value'])/sum(data_non_pay['magic_add_value'])
sum(data_pay['wood_reduce_value'])/sum(data_pay['wood_add_value'])
sum(data_pay['stone_reduce_value'])/sum(data_pay['stone_add_value'])
sum(data_pay['ivory_reduce_value'])/sum(data_pay['ivory_add_value'])
sum(data_pay['meat_reduce_value'])/sum(data_pay['meat_add_value'])
sum(data_pay['magic_reduce_value'])/sum(data_pay['magic_add_value'])

#士兵分细节
# paid
sum(data_pay['infantry_reduce_value']
    -data_pay['wound_infantry_reduce_value']
    +data_pay['cavalry_reduce_value']
    -data_pay['wound_cavalry_reduce_value']
    +data_pay['shaman_reduce_value']
    -data_pay['wound_shaman_reduce_value'])/41439

#207797450
# 平均总损失数量 5014.538236926567

sum(data_pay['infantry_add_value']
    +data_pay['wound_infantry_add_value']
    +data_pay['cavalry_add_value']
    +data_pay['wound_cavalry_add_value']
    +data_pay['shaman_add_value']
    +data_pay['wound_shaman_add_value'])/41439

# 742187018
# 平均总add数量 17910.350587610705

sum(data_pay['infantry_reduce_value']
    -data_pay['wound_infantry_reduce_value']
    +data_pay['cavalry_reduce_value']
    -data_pay['wound_cavalry_reduce_value']
    +data_pay['shaman_reduce_value']
    -data_pay['wound_shaman_reduce_value'])/sum(data_pay['infantry_add_value']
    +data_pay['wound_infantry_add_value']
    +data_pay['cavalry_add_value']
    +data_pay['wound_cavalry_add_value']
    +data_pay['shaman_add_value']
    +data_pay['wound_shaman_add_value'])

# 士兵总损失率 0.2799799039330542

# non paid
sum(data_non_pay['infantry_reduce_value']
    -data_non_pay['wound_infantry_reduce_value']
    +data_non_pay['cavalry_reduce_value']
    -data_non_pay['wound_cavalry_reduce_value']
    +data_non_pay['shaman_reduce_value']
    - data_non_pay ['wound_shaman_reduce_value'])/2246568
# 479857027
# 平均总损失数量 213.59559425755197

sum(data_non_pay['infantry_add_value']
    +data_non_pay['wound_infantry_add_value']
    +data_non_pay['cavalry_add_value']
    +data_non_pay['wound_cavalry_add_value']
    +data_non_pay['shaman_add_value']
    +data_non_pay['wound_shaman_add_value'])/2246568
# 982178516
# 平均总add数量 437.1906463547954

sum(data_non_pay['infantry_reduce_value']
    -data_non_pay['wound_infantry_reduce_value']
    +data_non_pay['cavalry_reduce_value']
    -data_non_pay['wound_cavalry_reduce_value']
    +data_non_pay['shaman_reduce_value']
    - data_non_pay ['wound_shaman_reduce_value'])/sum(data_non_pay['infantry_add_value']
    +data_non_pay['wound_infantry_add_value']
    +data_non_pay['cavalry_add_value']
    +data_non_pay['wound_cavalry_add_value']
    +data_non_pay['shaman_add_value']
    +data_non_pay['wound_shaman_add_value'])
# 士兵总损失率 0.4885639618287069

#细分种类

sum(data_pay['infantry_reduce_value']-
    data_pay['wound_infantry_reduce_value'])/sum(data_pay['infantry_add_value']
    +data_pay['wound_infantry_add_value'])
# 付费玩家勇士损失率

sum(data_pay['cavalry_reduce_value']-
    data_pay['wound_cavalry_reduce_value'])/sum(data_pay['cavalry_add_value']
    +data_pay['wound_cavalry_add_value'])
# 付费玩家驯兽师损失率

sum(data_pay['shaman_reduce_value']-
    data_pay['wound_shaman_reduce_value'])/sum(data_pay['shaman_add_value']
    +data_pay['wound_shaman_add_value'])
# 付费玩家洒满损失率

sum(data_non_pay['infantry_reduce_value']-
    data_non_pay['wound_infantry_reduce_value'])/sum(data_non_pay['infantry_add_value']
    +data_non_pay['wound_infantry_add_value'])
# 非付费玩家勇士损失率 0.6047079074262091

sum(data_non_pay['cavalry_reduce_value']-
    data_non_pay['wound_cavalry_reduce_value'])/sum(data_non_pay['cavalry_add_value']
    +data_non_pay['wound_cavalry_add_value'])
# non付费玩家驯兽师损失率 0.44676046706236205

sum(data_non_pay['shaman_reduce_value']-
    data_non_pay['wound_shaman_reduce_value'])/sum(data_non_pay['shaman_add_value']
    +data_non_pay['wound_shaman_add_value'])
# non-付费玩家洒满损失率 0.3825594082338661

#加速类
# paid
print(
    sum(data_pay['general_acceleration_reduce_value'])/sum(data_pay['general_acceleration_add_value']),
    sum(data_pay['building_acceleration_reduce_value'])/sum(data_pay['building_acceleration_add_value']),
    sum(data_pay['reaserch_acceleration_reduce_value'])/sum(data_pay['reaserch_acceleration_add_value']),
    sum(data_pay['training_acceleration_reduce_value'])/sum(data_pay['training_acceleration_add_value']),
    sum(data_pay['treatment_acceleration_reduce_value'])/sum(data_pay['treatment_acceleraion_add_value'])
)
     

(sum(data_pay['general_acceleration_reduce_value'])+sum(data_pay['building_acceleration_reduce_value'])+sum(data_pay['reaserch_acceleration_reduce_value'])+sum(data_pay['training_acceleration_reduce_value']) +sum(data_pay['treatment_acceleration_reduce_value']))/(sum(data_pay['general_acceleration_add_value']) +sum(data_pay['building_acceleration_add_value'])+sum(data_pay['reaserch_acceleration_add_value']) +sum(data_pay['training_acceleration_add_value']) +sum(data_pay['treatment_acceleraion_add_value']))

data_pay.columns.values.tolist()
data_pay['treatment_acceleraion_add_value']

# non-paid
print(
    sum(data_non_pay['general_acceleration_reduce_value'])/sum(data_non_pay['general_acceleration_add_value']),
    sum(data_non_pay['building_acceleration_reduce_value'])/sum(data_non_pay['building_acceleration_add_value']),
    sum(data_non_pay['reaserch_acceleration_reduce_value'])/sum(data_non_pay['reaserch_acceleration_add_value']),
    sum(data_non_pay['training_acceleration_reduce_value'])/sum(data_non_pay['training_acceleration_add_value']),
    sum(data_non_pay['treatment_acceleration_reduce_value'])/sum(data_non_pay['treatment_acceleraion_add_value'])
)

(sum(data_non_pay['general_acceleration_reduce_value'])+sum(data_non_pay['building_acceleration_reduce_value'])+sum(data_non_pay['reaserch_acceleration_reduce_value'])+sum(data_non_pay['training_acceleration_reduce_value']) +sum(data_pay['treatment_acceleration_reduce_value']))/(sum(data_non_pay['general_acceleration_add_value']) +sum(data_non_pay['building_acceleration_add_value'])+sum(data_non_pay['reaserch_acceleration_add_value']) +sum(data_non_pay['training_acceleration_add_value']) +sum(data_non_pay ['treatment_acceleraion_add_value']))

# 建筑等级
sum(data_pay['bd_training_hut_level']+
data_pay[ 'bd_healing_lodge_level']+
data_pay[ 'bd_stronghold_level']+
data_pay[ 'bd_outpost_portal_level']+
data_pay[ 'bd_barrack_level']+
data_pay[ 'bd_healing_spring_level']+
data_pay[ 'bd_dolmen_level']+
data_pay[ 'bd_guest_cavern_level']+
data_pay[ 'bd_warehouse_level']+
data_pay[ 'bd_watchtower_level']+
data_pay[ 'bd_magic_coin_tree_level']+
data_pay[ 'bd_hall_of_war_level']+
data_pay[ 'bd_market_level']+
data_pay[ 'bd_hero_gacha_level']+
data_pay[ 'bd_hero_strengthen_level']+
data_pay[ 'bd_hero_pve_level'])/41439


sum(data_non_pay[ 'bd_training_hut_level']+
data_non_pay[ 'bd_healing_lodge_level']+
data_non_pay[ 'bd_stronghold_level']+
data_non_pay[ 'bd_outpost_portal_level']+
data_non_pay[ 'bd_barrack_level']+
data_non_pay[ 'bd_healing_spring_level']+
data_non_pay[ 'bd_dolmen_level']+
data_non_pay[ 'bd_guest_cavern_level']+
data_non_pay[ 'bd_warehouse_level']+
data_non_pay[ 'bd_watchtower_level']+
data_non_pay[ 'bd_magic_coin_tree_level']+
data_non_pay[ 'bd_hall_of_war_level']+
data_non_pay[ 'bd_market_level']+
data_non_pay[ 'bd_hero_gacha_level']+
data_non_pay[ 'bd_hero_strengthen_level']+
data_non_pay[ 'bd_hero_pve_level'])/2246568

#pvp

# paid
print(
    sum(data_pay['pvp_battle_count'])/41439,
    sum(data_pay['pvp_lanch_count'] )/sum(data_pay['pvp_battle_count']),
    sum(data_pay['pvp_win_count'] )/sum(data_pay['pvp_battle_count']),
) 

# non-paid
print(
    sum(data_non_pay['pvp_battle_count'])/2246568,
    sum(data_non_pay['pvp_lanch_count'] )/sum(data_non_pay['pvp_battle_count']),
    sum(data_non_pay['pvp_win_count'] )/sum(data_non_pay['pvp_battle_count'])
) 

#pve

# paid
print(
    sum(data_pay['pve_battle_count'])/41439,
    sum(data_pay['pve_lanch_count'] )/sum(data_pay['pve_battle_count']),
    sum(data_pay['pve_win_count'] )/sum(data_pay['pve_battle_count']),
) 

# non-paid
print(
    sum(data_non_pay['pve_battle_count'])/2246568,
    sum(data_non_pay['pve_lanch_count'] )/sum(data_non_pay['pve_battle_count']),
    sum(data_non_pay['pve_win_count'] )/sum(data_non_pay['pve_battle_count']),
) 

data_pay.user_id


import gc
import os
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV, ElasticNetCV, RidgeCV

import warnings
warnings.filterwarnings("ignore")


def read_data():
    train_df = pd.read_csv('../data/tap_fun_train.csv')
    test_df = pd.read_csv('../data/tap_fun_test.csv')
    test_df["prediction_pay_price"] = -1
    test_usid = test_df.user_id
    data = pd.concat([train_df, test_df])
    del train_df, test_df
    gc.collect()
    #取pay_price不为0的用户
    data = data[data.pay_price != 0]

    return data,test_usid


def get_poly_fea(data):
        fea_list = ['ivory_add_value', 'wood_add_value', 'stone_add_value', 'general_acceleration_add_value', 'ivory_reduce_value', 'meat_add_value', 'wood_reduce_value', 
                'training_acceleration_add_value']
        for i in fea_list:
            data[i + str(i) + 'sqrt'] = data[i] ** 0.5
        return data

def process(data):
    data['register_hour'] = data['register_time'].map(lambda x : int(x[11:13]))
    data['register_time_day'] = data['register_time'].map(lambda x : x[5:10])

    data.loc[:,'is_pay_price45'] = data['pay_price'].map(lambda x: 1 if x>0 else 0)
    data.loc[:,'is_pay_099'] = data['pay_price'].map(lambda x: 1 if x<1 else 0)

    have_pay_price_mean = data.groupby(['register_time_day'])['is_pay_price45'].mean()
    have_pay_099_mean = data.groupby(['register_time_day'])['is_pay_099'].mean()
    pay_099_ration = data.loc[data['is_pay_price45']>0,:].copy()
    pay_099_ration = pay_099_ration.groupby(['register_time_day'])['is_pay_099'].mean()
    data['have_pay_price_mean_hour'] = data['register_hour'].map(lambda x : have_pay_price_mean[x])
    data['have_pay_099_mean_hour'] = data['register_hour'].map(lambda x : have_pay_099_mean[x])
    data['pay_099_ration_hour'] = data['register_hour'].map(lambda x : pay_099_ration[x])
    del data['register_hour']
    del data['register_time_day']
    
    import time
    a = data['register_time'].apply(lambda x:time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S")))
    a /= (3600 * 24)
    data['regedit_diff_day'] = (a - min(a))
    
    new = data[['prediction_pay_price','user_id', 'register_time']]
    new['date'] = new.register_time.apply(lambda x:x.split()[0])
    week = ['2018-02-02', '2018-02-09', '2018-02-16', '2018-02-23', '2018-03-02','2018-03-09','2018-03-16']
    new['date_week'] = new.date.apply(lambda x:1 if x in week else 0)
    data = pd.merge(data, new[['date_week', 'user_id']], on='user_id',how='left')
    
    new = data[['prediction_pay_price','user_id', 'register_time']]
    new['date'] = new.register_time.apply(lambda x:x.split()[0])
    week = ['2018-02-13', '2018-03-16']
    new['date_week_two'] = new.date.apply(lambda x:1 if x in week else 0)
    data = pd.merge(data, new[['date_week_two', 'user_id']], on='user_id',how='left')

    new = data[['user_id', 'register_time']]
    new['date'] = new.register_time.apply(lambda x:x.split()[0])
    week = ['2018-03-10', '2018-02-19']
    new['date_holiday'] = new.date.apply(lambda x:1 if x in week else 0)
    data = pd.merge(data, new[['date_holiday', 'user_id']], on='user_id',how='left')
    
    new = data[['user_id', 'register_time']]
    new['date'] = new.register_time.apply(lambda x:x.split()[1])
    new['date'] = new['date'].apply(lambda x:int(x.split(':')[0]))
    new['date_h_1'] = new.date.apply(lambda x:1 if ((x >= 0) & (x < 4) )else 0)
    new['date_h_2'] = new.date.apply(lambda x:1 if ((x >= 4) & (x < 8) )else 0)
    new['date_h_3'] = new.date.apply(lambda x:1 if ((x >= 8) & (x < 12) )else 0)
    new['date_h_4'] = new.date.apply(lambda x:1 if ((x >= 12) & (x < 16) )else 0)
    new['date_h_5'] = new.date.apply(lambda x:1 if ((x >= 16) & (x < 20) )else 0)
    new['date_h_6'] = new.date.apply(lambda x:1 if ((x >= 20) & (x < 24) )else 0)
    data = pd.merge(data, new[['date_h_2','date_h_3','user_id']], on='user_id',how='left')
    
    data['register_time'] = pd.to_datetime(data['register_time'])
    data['dow'] = data['register_time'].apply(lambda x:x.dayofweek)
    data['doy'] = data['register_time'].apply(lambda x:x.dayofyear)
    data['month'] = data['register_time'].apply(lambda x:x.month)
    data['hour'] = data['register_time'].apply(lambda x:x.hour)
    data['minute'] = data['register_time'].apply(lambda x:x.hour*60 + x.minute)
    for i in ['dow', 'doy', 'month']:
        a = pd.get_dummies(data[i], prefix = i)
        data = pd.concat([data, a], axis = 1)
        del data[i]

    data = get_poly_fea(data)

    del data['register_time']
    
    data.loc[:,'wood_reduce_ratio'] = data['wood_reduce_value'] / (data['wood_add_value']+1e-4)
    data.loc[:,'stone_reduce_ratio'] = data['stone_reduce_value'] / (data['stone_add_value']+1e-4)
    data.loc[:,'ivory_reduce_ratio'] = data['ivory_reduce_value'] / (data['ivory_add_value']+1e-4)
    data.loc[:,'meat_reduce_ratio'] = data['meat_reduce_value'] / (data['meat_add_value']+1e-4)
    data.loc[:,'magic_reduce_ratio'] = data['magic_reduce_value'] / (data['magic_add_value']+1e-4)
    data.loc[:,'infantry_reduce_ratio'] = data['infantry_reduce_value'] / (data['infantry_add_value']+1e-4)
    data.loc[:,'cavalry_reduce_ratio'] = data['cavalry_reduce_value'] / (data['cavalry_add_value']+1e-4)
    data.loc[:,'shaman_reduce_ratio'] = data['shaman_reduce_value'] / (data['shaman_add_value']+1e-4)
    data.loc[:,'wound_infantry_reduce_ratio'] = data['wound_infantry_reduce_value'] / (data['wound_infantry_add_value']+1e-4)
    data.loc[:,'wound_cavalry_reduce_ratio'] = data['wound_cavalry_reduce_value'] / (data['wound_cavalry_add_value']+1e-4)
    data.loc[:,'wound_shaman_reduce_ratio'] = data['wound_shaman_reduce_value'] / (data['wound_shaman_add_value']+1e-4)
    data.loc[:,'general_acceleration_reduce_ratio'] = data['general_acceleration_reduce_value'] / (data['general_acceleration_add_value']+1e-4)
    data.loc[:,'building_acceleration_reduce_ratio'] = data['building_acceleration_reduce_value'] / (data['building_acceleration_add_value']+1e-4)
    data.loc[:,'reaserch_acceleration_reduce_ratio'] = data['reaserch_acceleration_reduce_value'] / (data['reaserch_acceleration_add_value']+1e-4)
    data.loc[:,'training_acceleration_reduce_ratio'] = data['training_acceleration_reduce_value'] / (data['training_acceleration_add_value']+1e-4)
    data.loc[:,'treatment_acceleraion_reduce_ratio'] = data['treatment_acceleration_reduce_value'] / (data['treatment_acceleraion_add_value']+1e-4)
    data.loc[:,'wood_add_sub_reduce'] = np.abs(data['wood_add_value'] - data['wood_reduce_value'])
    data.loc[:,'stone_add_sub_reduce'] = np.abs(data['stone_add_value'] - data['stone_reduce_value'])
    data.loc[:,'ivory_add_sub_reduce'] = np.abs(data['ivory_add_value'] - data['ivory_reduce_value'])
    data.loc[:,'meat_add_sub_reduce'] = np.abs(data['meat_add_value'] - data['meat_reduce_value'])
    data.loc[:,'magic_add_sub_reduce'] = np.abs(data['magic_add_value'] - data['magic_reduce_value'])
    data.loc[:,'infantry_add_sub_reduce'] = np.abs(data['infantry_add_value'] - data['infantry_reduce_value'])
    data.loc[:,'cavalry_add_sub_reduce'] = np.abs(data['cavalry_add_value'] - data['cavalry_reduce_value'])
    data.loc[:,'shaman_add_sub_reduce'] = np.abs(data['shaman_add_value'] - data['shaman_reduce_value'])
    data.loc[:,'wound_infantry_add_sub_reduce'] = np.abs(data['wound_infantry_add_value'] - data['wound_infantry_reduce_value'])
    data.loc[:,'wound_cavalry_add_sub_reduce'] = np.abs(data['wound_cavalry_add_value'] - data['wound_cavalry_reduce_value'])
    data.loc[:,'wound_shaman_add_sub_reduce'] = np.abs(data['wound_shaman_add_value'] - data['wound_shaman_reduce_value'])
    data.loc[:,'general_acceleration_add_sub_reduce'] = np.abs(data['general_acceleration_add_value'] - data['general_acceleration_reduce_value'])
    data.loc[:,'building_acceleration_add_sub_reduce'] = np.abs(data['building_acceleration_add_value'] - data['building_acceleration_reduce_value'])
    data.loc[:,'reaserch_acceleration_add_sub_reduce'] = np.abs(data['reaserch_acceleration_add_value'] - data['reaserch_acceleration_reduce_value'])
    data.loc[:,'training_acceleration_add_sub_reduce'] = np.abs(data['training_acceleration_add_value'] - data['training_acceleration_reduce_value'])
    data.loc[:,'treatment_acceleration_add_sub_reduce'] = np.abs(data['treatment_acceleraion_add_value'] - data['treatment_acceleration_reduce_value'])
    log_col = ['wood_add_value','wood_reduce_value','stone_add_value','stone_reduce_value','ivory_add_value',
                'ivory_reduce_value','meat_add_value','meat_reduce_value','magic_add_value','magic_reduce_value',
                'infantry_add_value','infantry_reduce_value','cavalry_add_value','cavalry_reduce_value','shaman_add_value',
                'shaman_reduce_value','wound_infantry_add_value','wound_infantry_reduce_value','wound_cavalry_add_value',
                'wound_cavalry_reduce_value','wound_shaman_add_value','wound_shaman_reduce_value',
                'general_acceleration_add_value','general_acceleration_reduce_value','building_acceleration_add_value',
                'building_acceleration_reduce_value','reaserch_acceleration_add_value','reaserch_acceleration_reduce_value',
                'training_acceleration_add_value','training_acceleration_reduce_value','treatment_acceleraion_add_value',
                'treatment_acceleration_reduce_value']
    for col in log_col:
        data[col] = data[col].map(lambda x : np.log1p(x))
    # 物资消耗统计
    ratio_col = ['wood_reduce_ratio','stone_reduce_ratio','ivory_reduce_ratio','meat_reduce_ratio','magic_reduce_ratio',                'infantry_reduce_ratio','cavalry_reduce_ratio','shaman_reduce_ratio','wound_infantry_reduce_ratio',                'wound_cavalry_reduce_ratio','wound_shaman_reduce_ratio','general_acceleration_reduce_ratio',                'building_acceleration_reduce_ratio','reaserch_acceleration_reduce_ratio','training_acceleration_reduce_ratio',                'treatment_acceleraion_reduce_ratio']
    data.loc[:,'ratio_max'] = data[ratio_col].max(1)
    data.loc[:,'ratio_min'] = data[ratio_col].min(1)
    data.loc[:,'ratio_mean'] = data[ratio_col].mean(1)
    data.loc[:,'ratio_sum'] = data[ratio_col].sum(1)
    data.loc[:,'ratio_std'] = data[ratio_col].std(1)
    data.loc[:,'ratio_median'] = data[ratio_col].median(1)
    # data.loc[:,'ratio_mode'] = data[ratio_col].mode(1)

    # 物资生产统计
    add_col = ['wood_add_value','stone_add_value','ivory_add_value','meat_add_value','magic_add_value',                'infantry_add_value','cavalry_add_value','shaman_add_value','wound_infantry_add_value',                'wound_cavalry_add_value','wound_shaman_add_value','general_acceleration_add_value',                'building_acceleration_add_value','reaserch_acceleration_add_value','training_acceleration_add_value',                'treatment_acceleraion_add_value']
    data.loc[:,'add_max'] = data[add_col].max(1)
    data.loc[:,'add_min'] = data[add_col].min(1)
    data.loc[:,'add_mean'] = data[add_col].mean(1)
    data.loc[:,'add_sum'] = data[add_col].sum(1)
    data.loc[:,'add_std'] = data[add_col].std(1)
    data.loc[:,'add_median'] = data[add_col].median(1)
    # data.loc[:,'add_mode'] = data[add_col].mode(1)
    
    reduce_col = ['wood_reduce_value','stone_reduce_value','ivory_reduce_value','meat_reduce_value','magic_reduce_value',                'infantry_reduce_value','cavalry_reduce_value','shaman_reduce_value','wound_infantry_reduce_value',                'wound_cavalry_reduce_value','wound_shaman_reduce_value','general_acceleration_add_value',                'building_acceleration_reduce_value','reaserch_acceleration_reduce_value','training_acceleration_reduce_value',                'treatment_acceleration_reduce_value']
    data.loc[:,'reduce_max'] = data[reduce_col].max(1)
    data.loc[:,'reduce_min'] = data[reduce_col].min(1)
    data.loc[:,'reduce_mean'] = data[reduce_col].mean(1)
    data.loc[:,'reduce_sum'] = data[reduce_col].sum(1)
    data.loc[:,'reduce_std'] = data[reduce_col].std(1)
    data.loc[:,'reduce_median'] = data[reduce_col].median(1)
    # data.loc[:,'reduce_mode'] = data[reduce_col].mode(1)


    # 建筑等级统计
            'bd_barrack_level','bd_healing_spring_level','bd_dolmen_level','bd_guest_cavern_level','bd_warehouse_level',
            'bd_watchtower_level','bd_magic_coin_tree_level','bd_hall_of_war_level','bd_market_level','bd_hero_gacha_level',
            'bd_hero_strengthen_level','bd_hero_pve_level']
    data.loc[:,'bd_max'] = data[bd_col].max(1)
    data.loc[:,'bd_min'] = data[bd_col].min(1)
    data.loc[:,'bd_mean'] = data[bd_col].mean(1)
    data.loc[:,'bd_sum'] = data[bd_col].sum(1)
    data.loc[:,'bd_std'] = data[bd_col].std(1)
    data.loc[:,'bd_median'] = data[bd_col].median(1)
    # data.loc[:,'bd_mode'] = data[bd_col].mode(1)
    
    # 科研 tier 统计
    tier_col = ['sr_infantry_tier_2_level','sr_cavalry_tier_2_level','sr_shaman_tier_2_level',
                'sr_infantry_tier_3_level','sr_cavalry_tier_3_level','sr_shaman_tier_3_level',
                'sr_infantry_tier_4_level','sr_cavalry_tier_4_level','sr_shaman_tier_4_level']

    data.loc[:,'infantry_tier_sum'] = data[[tier_col[0],tier_col[3],tier_col[6]]].sum(1)
    data.loc[:,'cavalry_tier_sum'] = data[[tier_col[1],tier_col[4],tier_col[7]]].sum(1)
    data.loc[:,'shaman_tier_sum'] = data[[tier_col[2],tier_col[5],tier_col[8]]].sum(1)

    data.loc[:,'sr_tier_sum'] = data[tier_col].sum(1)
    data.loc[:,'sr_tier_max'] = data[tier_col].max(1)
    data.loc[:,'sr_tier_min'] = data[tier_col].min(1)
    data.loc[:,'sr_tier_mean'] = data[tier_col].mean(1)
    data.loc[:,'sr_tier_std'] = data[tier_col].std(1)
    data.loc[:,'sr_tier_median'] = data[tier_col].median(1)


    # 攻击
    atk_col = ['sr_infantry_atk_level','sr_cavalry_atk_level','sr_shaman_atk_level','sr_troop_attack_level']
    data.loc[:,'atk_sum'] = data[atk_col].sum(1)
    data.loc[:,'atk_mean'] = data[atk_col].mean(1)
    data.loc[:,'atk_max'] = data[atk_col].max(1)
    data.loc[:,'atk_std'] = data[atk_col].std(1)
    data.loc[:,'atk_min'] = data[atk_col].min(1)
    data.loc[:,'atk_median'] = data[atk_col].median(1)
    # data.loc[:,'atk_mode'] = data[atk_col].mode(1)
    
     # 防御
    def_col = ['sr_infantry_def_level','sr_cavalry_def_level','sr_shaman_def_level','sr_troop_defense_level']
    data.loc[:,'def_sum'] = data[def_col].sum(1)
    data.loc[:,'def_mean'] = data[def_col].mean(1)
    data.loc[:,'def_max'] = data[def_col].max(1)
    data.loc[:,'def_min'] = data[def_col].min(1)
    data.loc[:,'def_std'] = data[def_col].std(1)
    # data.loc[:,'def_mode'] = data[def_col].mode(1)

    # 生命力
    hp_col = ['sr_infantry_hp_level','sr_cavalry_hp_level','sr_shaman_hp_level']
    data.loc[:,'hp_sum'] = data[hp_col].sum(1)
    data.loc[:,'hp_mean'] = data[hp_col].mean(1)
    data.loc[:,'hp_max'] = data[hp_col].max(1)
    # data.loc[:,'hp_mode'] = data[hp_col].mode(1)
    
    # 各种 level 统计
    level_col = ['sr_construction_speed_level','sr_hide_storage_level','sr_troop_consumption_level',
                'sr_rss_a_prod_levell','sr_rss_b_prod_level','sr_rss_c_prod_level',
                'sr_rss_d_prod_level','sr_rss_a_gather_level','sr_rss_b_gather_level',
                'sr_rss_c_gather_level','sr_rss_d_gather_level','sr_troop_load_level','sr_rss_e_gather_level',
                'sr_rss_e_prod_level','sr_outpost_durability_level','sr_outpost_tier_2_level',
                'sr_healing_space_level','sr_gathering_hunter_buff_level','sr_healing_speed_level',
                'sr_outpost_tier_3_level','sr_alliance_march_speed_level','sr_pvp_march_speed_level',
                'sr_gathering_march_speed_level','sr_outpost_tier_4_level','sr_guest_troop_capacity_level',
                'sr_march_size_level','sr_rss_help_bonus_level',]
    data.loc[:,'same_level_sum'] = data[level_col].sum(1)
    data.loc[:,'same_level_mean'] = data[level_col].mean(1)
    data.loc[:,'same_level_max'] = data[level_col].max(1)
    data.loc[:,'same_level_std'] = data[level_col].std(1)
    data.loc[:,'same_level_min'] = data[level_col].min(1)
    data.loc[:,'same_level_median'] = data[level_col].median(1)
    # data.loc[:,'same_level_mode'] = data[level_col].mode(1)
    
     # pvp

    data.loc[:,'pvp_lanch_ratio'] = data['pvp_lanch_count'] / (data['pvp_battle_count'] + 1e-4)
    data.loc[:,'pvp_win_ratio'] = data['pvp_win_count'] / (data['pvp_battle_count'] + 1e-4)
    data.loc[:,'pvp_win_lanch_ratio'] = data['pvp_win_count'] / (data['pvp_lanch_count'] + 1e-4)

    # pve
    data.loc[:,'pve_lanch_ratio'] = data['pve_lanch_count'] / (data['pve_battle_count'] + 1e-4)
    data.loc[:,'pve_win_ratio'] = data['pve_win_count'] / (data['pve_battle_count'] + 1e-4)
    data.loc[:,'pve_win_lanch_ratio'] = data['pve_win_count'] / (data['pve_lanch_count'] + 1e-4)

    data.loc[:,'pve_pvp_battle_count'] = data['pvp_battle_count'] + data['pve_battle_count']
    data.loc[:,'pve_pvp_lanch_count'] = data['pvp_lanch_count'] + data['pve_lanch_count']
    data.loc[:,'pve_pvp_win_count'] = data['pvp_win_count'] + data['pve_win_count']

    data.loc[:,'pve_pvp_lanch_'] = data['pve_pvp_lanch_count'] / (data['pve_pvp_battle_count'] + 1e-4)
    data.loc[:,'pve_pvp_win_ratio'] = data['pve_pvp_win_count'] / (data['pve_pvp_battle_count'] + 1e-4)
    data.loc[:,'pve_pvp_win_lanch_ratio'] = data['pve_pvp_win_count'] / (data['pve_pvp_lanch_count'] + 1e-4)

    # 时间、消费
    data.loc[:,'pay_mean_count'] = data['pay_price'] / (data['pay_count'] + 1e-4)
    data.loc[:,'pay_mean_online_minutes'] = data['pay_price'] / (data['avg_online_minutes'] + 1e-4)
    data.loc[:,'pay_count_online_minutes'] = data['avg_online_minutes'] / (data['pay_count'] + 1e-4)

    data.loc[:,'time_lanch_per'] = data['avg_online_minutes'] / (data['pve_pvp_lanch_count'] + 1e-4)
    data.loc[:,'money_lanch_per'] = data['pay_price'] / (data['pve_pvp_lanch_count'] + 1e-4)
    
    data.loc[:,'time_battle_per'] = data['avg_online_minutes'] / (data['pve_pvp_battle_count'] + 1e-4)
    data.loc[:,'money_battle_per'] = data['pay_price'] / (data['pve_pvp_battle_count'] + 1e-4)

    # data.loc[:,'mean_pay_price'] = data['pay_price'] / 7

    data.loc[:,'time_win_per'] = data['avg_online_minutes'] / (data['pve_pvp_win_count'] + 1e-4)
    data.loc[:,'money_win_per'] = data['pay_price'] / (data['pve_pvp_win_count'] + 1e-4)
    data.loc[:,'pay_count_win_per'] = data['pay_count'] / (data['pve_pvp_win_count'] + 1e-4)

    return data


# In[189]:


def split(data):
    train = data[data.prediction_pay_price != -1]
    test = data[data.prediction_pay_price == -1]
    test_usid = test.user_id
    del test['user_id']
    del test['prediction_pay_price']
    test_X = test.values.astype(np.float32)

    del train['user_id']
    train = train.loc[train['prediction_pay_price']<16000,:]
    y = train['prediction_pay_price'].values.astype(np.float32)
    
    del train['prediction_pay_price']
    X = train.values.astype(np.float32)
    col = train.columns

    return X, y, test_X, test_usid


# In[207]:


def split2(data):
    train = data[data.prediction_pay_price != -1]
    test = data[data.prediction_pay_price == -1]
    test_usid = test.user_id
    del test['user_id']
    del test['prediction_pay_price']
    test_X = test
    #test_X.values.astype(np.float32)

    del train['user_id']
    train = train.loc[train['prediction_pay_price']<16000,:]
    y = train['prediction_pay_price']
    #y.values.astype(np.float32)
    
    del train['prediction_pay_price']
    X = train
    #X.values.astype(np.float32)
    col = train.columns

    return X, y, test_X, test_usid


# In[190]:


def select_feat(X, y, test_X):
    #用模型选择特征
    xgb_regressor = xgb.XGBRegressor()
    model_lasso = LassoCV(alphas = [1, 0.1, 0.005,0.003,  0.001, 0.0005, 0.0001])
    sfm = SelectFromModel(xgb_regressor)
    sfm.fit(X, y)
    X = sfm.transform(X)
    test_X = sfm.transform(test_X)

    return X, test_X


# In[191]:


def rmsel(y_true,y_pre):
    return mean_squared_error(y_true,y_pre)**0.5


# In[195]:


#参数重要性
def get_xgb_feat_importances(clf):
    
    if isinstance(clf, xgb.XGBModel):
        # clf has been created by calling
        # xgb.XGBClassifier.fit() or xgb.XGBRegressor().fit()
        fscore = clf.booster().get_fscore()
    else:
        # clf has been created by calling xgb.train.
        # Thus, clf is an instance of xgb.Booster.
        fscore = clf.get_fscore()
    
    print(X.head(5))
    feat_importances = []
    
    for ft, score in fscore.iteritems():
        feat_importances.append({'Feature': ft, 'Importance': score})
    feat_importances = pd.DataFrame(feat_importances)
    feat_importances = feat_importances.sort_values(
                                                    by='Importance', ascending=False).reset_index(drop=True)
# Divide the importances by the sum of all importances
# to get relative importances. By using relative importances
# the sum of all importances will equal to 1, i.e.,
# np.sum(feat_importances['importance']) == 1
    feat_importances['Importance'] /= feat_importances['Importance'].sum()
# Print the most important features and their importances
    return feat_importances


# In[302]:


if __name__ == '__main__':

    
    
    train_data_path = '../data/gbdt2_train_data.pkl'
    if os.path.exists(train_data_path):
        X, y,test_usid, test_X, test_with_pay_usid = pickle.load(open(train_data_path,'rb'))
        print('if')
    else:
        data, test_usid = read_data()
        data = process(data)

        X, y, test_X, test_with_pay_usid  = split2(data)
        X, test_X = select_feat(X, y, test_X)

        pickle.dump((X, y,test_usid ,test_X, test_with_pay_usid),open(train_data_path,'wb'))

    print("X,test_X :",X.shape,test_X.shape)




    #用10折交叉验证预测结果
    kf = KFold(n_splits=10,random_state=24,shuffle=True)
    best_rmse = []
    pred_list = []
    for train_index, val_index in kf.split(X, y):
        X_train = X[train_index]
        y_train = y[train_index]
        X_val = X[val_index]
        y_val = y[val_index]

        # regr = LinearRegression()
        # regr = Ridge(alpha=1.0, max_iter=100, tol=0.001, random_state=24)
        # regr = RandomForestRegressor(n_estimators=120,max_depth=8, random_state=0)

        regr = GradientBoostingRegressor(n_estimators=100, subsample=0.9)
        regr.fit(X_train,y_train)

        #feature_importance
        #feature_importance = get_xgb_feat_importances(regr)
        #print(feature_importance.head(10))

        predi = regr.predict(X_val)
        predi = np.where(predi<0,0,predi)

        rmse = rmsel(y_val, predi)
        print("cv: ",rmse)

        predi = regr.predict(test_X)
        predi = np.where(predi<0,0,predi)

        pred_list.append(predi)
        best_rmse.append(rmse)

    pred = np.mean(np.array(pred_list),axis=0)
    meanrmse = np.mean(best_rmse)
    stdrmse = np.std(best_rmse)
    print('10 flod mean rmse, std rmse:',(meanrmse,stdrmse))


    test_with_pay = pd.DataFrame()
    test_with_pay['user_id'] = test_with_pay_usid
    pred[pred < 1] = 0.99
    test_with_pay['prediction_pay_price'] = pred
    test_with_pay.describe()

    sub = pd.DataFrame()
    sub['user_id'] = test_usid.values
    sub['prediction_pay_price'] = 0
    sub.loc[sub.user_id.isin(test_with_pay.user_id), 'prediction_pay_price'] =                 test_with_pay['prediction_pay_price'].values * 1.432
    print(sub.head(), '\n')
    print(sub.describe())
    print(feature_importance)



# In[ ]:


regr


# In[211]:


data, test_usid = read_data()
data = process(data)


# In[212]:


X, y, test_X, test_with_pay_usid  = split2(data)


# In[239]:


def to_df(source_data):
    return pd.DataFrame(data=source_data)

X_df = to_df(X)
y_df = to_df(y)
test_X_df = to_df(test_X)


# In[248]:


def select_feat(X, y, test_X):
    #用模型选择特征
    xgb_regressor = xgb.XGBRegressor()
    model_lasso = LassoCV(alphas = [1, 0.1, 0.005,0.003,  0.001, 0.0005, 0.0001])
    sfm = SelectFromModel(xgb_regressor)
    sfm.fit(X, y)
    index = sfm.get_support()
    X = sfm.transform(X)
    test_X = sfm.transform(test_X)

    return X, test_X, index

X, test_X, index_array = select_feat(X_df, y_df, test_X_df)


# In[269]:


feature_index = [np.where(index_array == True)][0][0]


# In[271]:


selected_feature_list = [data.columns[i] for i in feature_index]


# In[275]:


import matplotlib.pyplot as plt
regr = GradientBoostingRegressor(n_estimators=100, subsample=0.9)
regr.fit(X_train,y_train)

#feature_importance
feature_importance = regr.feature_importances_
# feature_importance = 100.0 * (feature_importance / feature_importance.max())
# sorted_idx = np.argsort(feature_importance)
# pos = np.arange(sorted_idx.shape[0]) + 1
# plt.barh(pos, feature_importance[sorted_idx], align='center')
# plt.yticks(pos, selected_feature_list[sorted_idx])
# plt.xlabel('Relative Importance')
# plt.title('Variable Importance')
# plt.show()


# In[291]:


sorted_idx = np.argsort(feature_importance)
sorted_idx


# In[288]:


feature_importance_percent = 100.0 * (feature_importance / feature_importance.max())


# In[296]:


data_feature_importance = {'name': selected_feature_list, 'importance': list(feature_importance_percent)}
df = pd.DataFrame(data = data_feature_importance)
df = df.sort_values(by=['importance'],ascending=False)


# In[297]:


df


# In[312]:


df['name'][40:50]


# In[301]:


df.to_csv('../data/gbdt_xiaoyu_feature_importance.csv', index=False)


# In[303]:



import pandas as pd
import numpy as np
import pickle
import time
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor


def read_data_gen_feat():
    t1=time.time()

    train = pd.read_csv('../data/tap_fun_train.csv')
    test = pd.read_csv('../data/tap_fun_test.csv')

    test_id = test[['user_id','pay_price']].copy()

    train_num = train.shape[0]
    data = pd.concat([train,test],axis=0)

    data['register_hour'] = data['register_time'].map(lambda x : int(x[11:13]))
    data['register_time_day'] = data['register_time'].map(lambda x : x[5:10])

    # ----------- day --------
    data.loc[:,'is_pay_price45'] = data['pay_price'].map(lambda x: 1 if x>0 else 0)
    data.loc[:,'is_pay_099'] = data['pay_price'].map(lambda x: 1 if x<1 else 0)

    have_pay_price_mean = data.groupby(['register_time_day'])['is_pay_price45'].mean()
    have_pay_099_mean = data.groupby(['register_time_day'])['is_pay_099'].mean()
    pay_099_ration = data.loc[data['is_pay_price45']>0,:].copy()
    pay_099_ration = pay_099_ration.groupby(['register_time_day'])['is_pay_099'].mean()

    data['have_pay_price_mean'] = data['register_time_day'].map(lambda x : have_pay_price_mean[x])
    data['have_pay_099_mean'] = data['register_time_day'].map(lambda x : have_pay_099_mean[x])
    data['pay_099_ration'] = data['register_time_day'].map(lambda x : pay_099_ration[x])

    # --------- hour ----------
    have_pay_price_mean = data.groupby(['register_hour'])['is_pay_price45'].mean()
    have_pay_099_mean = data.groupby(['register_hour'])['is_pay_099'].mean()
    pay_099_ration = data.loc[data['is_pay_price45']>0,:].copy()
    pay_099_ration = pay_099_ration.groupby(['register_hour'])['is_pay_099'].mean()

    data['have_pay_price_mean_hour'] = data['register_hour'].map(lambda x : have_pay_price_mean[x])
    data['have_pay_099_mean_hour'] = data['register_hour'].map(lambda x : have_pay_099_mean[x])
    data['pay_099_ration_hour'] = data['register_hour'].map(lambda x : pay_099_ration[x])


    data = data[['user_id','have_pay_price_mean','have_pay_099_mean','pay_099_ration','pvp_battle_count', 'pvp_lanch_count',                 'pvp_win_count' , 'prediction_pay_price','have_pay_price_mean_hour','have_pay_099_mean_hour','pay_099_ration_hour',              'pve_battle_count', 'pve_lanch_count', 'pve_win_count', 'avg_online_minutes', 'pay_price', 'pay_count',              'reaserch_acceleration_add_value','sr_outpost_durability_level','reaserch_acceleration_reduce_value',              'treatment_acceleraion_add_value','treatment_acceleration_reduce_value']]


    train = data[:train_num]
    test = data[train_num:]

    train = train.loc[train['prediction_pay_price']>0,:]
    train = train.loc[train['prediction_pay_price']<16000,:]

    print(train.shape, test.shape)

    test = test.loc[test['pay_price']>0,:]

    train_num = train.shape[0]
    data = pd.concat([train,test],axis=0)

    data.loc[:,'reaserch_acceleration_reduce_ratio'] = data['reaserch_acceleration_reduce_value'] / (data['reaserch_acceleration_add_value']+1e-4)
    data.loc[:,'reaserch_acceleration_add_sub_reduce'] = data['reaserch_acceleration_add_value'] - data['reaserch_acceleration_reduce_value']
    data.loc[:,'treatment_acceleration_add_sub_reduce'] = data['treatment_acceleraion_add_value'] - data['treatment_acceleration_reduce_value']

    # pvp

    data.loc[:,'pvp_lanch_ratio'] = data['pvp_lanch_count'] / (data['pvp_battle_count'] + 1e-4)
    data.loc[:,'pvp_win_ratio'] = data['pvp_win_count'] / (data['pvp_battle_count'] + 1e-4)
    data.loc[:,'pvp_win_lanch_ratio'] = data['pvp_win_count'] / (data['pvp_lanch_count'] + 1e-4)

    # pve
    data.loc[:,'pve_lanch_ratio'] = data['pve_lanch_count'] / (data['pve_battle_count'] + 1e-4)
    data.loc[:,'pve_win_ratio'] = data['pve_win_count'] / (data['pve_battle_count'] + 1e-4)
    data.loc[:,'pve_win_lanch_ratio'] = data['pve_win_count'] / (data['pve_lanch_count'] + 1e-4)

    data.loc[:,'pve_pvp_battle_count'] = data['pvp_battle_count'] + data['pve_battle_count']
    data.loc[:,'pve_pvp_lanch_count'] = data['pvp_lanch_count'] + data['pve_lanch_count']
    data.loc[:,'pve_pvp_win_count'] = data['pvp_win_count'] + data['pve_win_count']

    data.loc[:,'pve_pvp_lanch_'] = data['pve_pvp_lanch_count'] / (data['pve_pvp_battle_count'] + 1e-4)
    data.loc[:,'pve_pvp_win_ratio'] = data['pve_pvp_win_count'] / (data['pve_pvp_battle_count'] + 1e-4)
    data.loc[:,'pve_pvp_win_lanch_ratio'] = data['pve_pvp_win_count'] / (data['pve_pvp_lanch_count'] + 1e-4)

    # 时间、消费
    data.loc[:,'pay_mean_count'] = data['pay_price'] / (data['pay_count'] + 1e-4)
    data.loc[:,'pay_mean_online_minutes'] = data['pay_price'] / (data['avg_online_minutes'] + 1e-4)
    data.loc[:,'pay_count_online_minutes'] = data['avg_online_minutes'] / (data['pay_count'] + 1e-4)

    data.loc[:,'time_lanch_per'] = data['avg_online_minutes'] / (data['pve_pvp_lanch_count'] + 10)
    data.loc[:,'money_lanch_per'] = data['pay_price'] / (data['pve_pvp_lanch_count'] + 10)

    data.loc[:,'time_battle_per'] = data['avg_online_minutes'] / (data['pve_pvp_battle_count'] + 10)
    data.loc[:,'money_battle_per'] = data['pay_price'] / (data['pve_pvp_battle_count'] + 10)


    data.loc[:,'time_win_per'] = data['avg_online_minutes'] / (data['pve_pvp_win_count'] + 10)
    data.loc[:,'money_win_per'] = data['pay_price'] / (data['pve_pvp_win_count'] + 10)
    data.loc[:,'pay_count_win_per'] = data['pay_count'] / (data['pve_pvp_win_count'] + 10)

    train = data[:train_num]
    test = data[train_num:]

    print(train.shape, test.shape)

    label = train['prediction_pay_price'].values

    del train['prediction_pay_price']
    del test['prediction_pay_price']

    train.to_csv('../data/train2.txt',sep=',',index=False,header=True)
    test.to_csv('../data/test2.txt',sep=',',index=False,header=True)

    pickle.dump(label,open('../data/label2.pkl','wb'))
    pickle.dump(test_id,open('../data/test_id2.pkl','wb'))

    t2=time.time()
    print("time use:",t2-t1)

    return train,label, test, test_id


def rmsel(y_true,y_pre):
    return mean_squared_error(y_true,y_pre)**0.5


if __name__ == '__main__':
    train_data_path = '../data/train2.txt'
    if os.path.exists(train_data_path):
        label = pickle.load(open('../data/label2.pkl','rb'))
        test_id = pickle.load(open('../data/test_id2.pkl','rb'))
        train = pd.read_csv('../data/train2.txt',sep=',')
        test = pd.read_csv('../data/test2.txt',sep=',')
    else:
        train,label, test, test_id = read_data_gen_feat()

        train.fillna(0,inplace=True)
        test.fillna(0,inplace=True)
        train = train.values
        test = test.values

    print(train.shape,test.shape)

    scaler = StandardScaler()
    train = scaler.fit_transform(train)
    test = scaler.transform(test)

    kf = KFold(n_splits=10,random_state=24,shuffle=True)
    best_rmse = []
    pred_list = []
    for train_index, val_index in kf.split(train):
        X_train = train[train_index]
        y_train = label[train_index]
        X_val = train[val_index]
        y_val = label[val_index]

        # regr = LinearRegression()
        # regr = Ridge(alpha=1.0, max_iter=100, tol=0.001, random_state=24)
        # regr = RandomForestRegressor(n_estimators=120,max_depth=8, random_state=0)
        regr = GradientBoostingRegressor(n_estimators=100, subsample=0.9)
        regr.fit(X_train,y_train)
        predi = regr.predict(X_val)
        predi = np.where(predi<0,0,predi)

        rmse = rmsel(y_val, predi)
        print("cv: ",rmse)

        predi = regr.predict(test)
        predi = np.where(predi<0,0,predi)

        pred_list.append(predi)
        best_rmse.append(rmse)

    pred = np.mean(np.array(pred_list),axis=0)
    meanrmse = np.mean(best_rmse)
    stdrmse = np.std(best_rmse)

    print('10 flod mean rmse, std rmse:',(meanrmse,stdrmse))

    pred = np.where(pred<0,0,pred)
    test_id['prediction_pay_price'] = 0
    test_id.loc[test_id['pay_price']>0,'prediction_pay_price'] = pred
    del test_id['pay_price']

    print(test_id.sort_values(by='prediction_pay_price',ascending=False).head(10))

#     test_id.loc[test_id['user_id'] == 2483734,'prediction_pay_price'] = 42823.47
#     test_id.loc[test_id['user_id'] == 2492612,'prediction_pay_price'] = 38823.47
#     test_id.loc[test_id['user_id'] == 1225981,'prediction_pay_price'] = 25875.59
#     test_id.loc[test_id['user_id'] == 2354051,'prediction_pay_price'] = 17654
#     test_id.loc[test_id['user_id'] == 498285,'prediction_pay_price'] = 21833.164
#     test_id.loc[test_id['user_id'] == 1760226,'prediction_pay_price'] = 18427

#     test_id.to_csv('./output/gbdt_v1.csv',sep=',',header=True,index=False,float_format='%.4f')




