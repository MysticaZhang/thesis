import pandas as pd

database= pd.read_csv('tap_fun_train.csv')
df = database[database['pay_price']>0]
database[database['pay_price']>0].count()


import matplotlib.pyplot as plt
import numpy as np

#在线时间 重肝与氪金
plt.scatter(df['avg_online_minutes'], df['pay_price'],  color='black')
plt.yscale('log')
plt.show()

#胜率 满足与付费
plt.scatter(df['pvp_win_count']/df['pvp_battle_count'], df['pay_price'],  color='black')
#plt.yscale('log')
#plt.xscale('log')

plt.show()

#拥有的资源与付费
plt.scatter(
    (df['bd_training_hut_level']+
    df['bd_healing_lodge_level']+
    df['bd_stronghold_level']+
    df['bd_outpost_portal_level']+
    df['bd_barrack_level']+
    df['bd_healing_spring_level']+
    df['bd_dolmen_level']+
    df['bd_guest_cavern_level']+
    df['bd_warehouse_level']+
    df['bd_watchtower_level']+
    df['bd_magic_coin_tree_level']+
    df['bd_hall_of_war_level']+
    df['bd_market_level']+
    df['bd_hero_gacha_level']+
    df['bd_hero_strengthen_level']+
    df['bd_hero_pve_level']+
    df['sr_scout_level']+
    df['sr_training_speed_level']+
    df['sr_infantry_tier_2_level']+
    df['sr_cavalry_tier_2_level']+
    df['sr_shaman_tier_2_level']+
    df['sr_infantry_atk_level']+
    df['sr_cavalry_atk_level']+
    df['sr_shaman_atk_level']+
    df['sr_infantry_tier_3_level']+
    df['sr_cavalry_tier_3_level']+
    df['sr_shaman_tier_3_level']+
    df['sr_troop_defense_level']+
    df['sr_infantry_def_level']+
    df['sr_cavalry_def_level']+
    df['sr_shaman_def_level']+
    df['sr_infantry_hp_level']+
    df['sr_cavalry_hp_level']+
    df['sr_shaman_hp_level']+
    df['sr_infantry_tier_4_level']+
    df['sr_cavalry_tier_4_level']+
    df['sr_shaman_tier_4_level']+
    df['sr_troop_attack_level']+
    df['sr_construction_speed_level']+
    df['sr_hide_storage_level']+
    df['sr_troop_consumption_level']+
    df['sr_rss_a_prod_levell']+
    df['sr_rss_b_prod_level']+
    df['sr_rss_c_prod_level']+
    df['sr_rss_d_prod_level']+
    df['sr_rss_a_gather_level']+
    df['sr_rss_b_gather_level']+
    df['sr_rss_c_gather_level']+
    df['sr_rss_d_gather_level']+
    df['sr_troop_load_level']+
    df['sr_rss_e_gather_level']+
    df['sr_rss_e_prod_level']+
    df['sr_outpost_durability_level']+
    df['sr_outpost_tier_2_level']+
    df['sr_healing_space_level']+
    df['sr_gathering_hunter_buff_level']+
    df['sr_healing_speed_level']+
    df['sr_outpost_tier_3_level']+
    df['sr_alliance_march_speed_level']+
    df['sr_pvp_march_speed_level']+
    df['sr_gathering_march_speed_level']+
    df['sr_outpost_tier_4_level']+
    df['sr_guest_troop_capacity_level']+
    df['sr_march_size_level']+
    df['sr_rss_help_bonus_level'])
    , df['pay_price'],  color='black')
#plt.yscale('log')
plt.xscale('log')

plt.show()

#资源使用率
plt.scatter(
        df['wood_reduce_value']+
        df['stone_reduce_value']+
        df['ivory_reduce_value']+
        df['meat_reduce_value']+
        df['magic_reduce_value']+
        df['infantry_reduce_value']+
        df['cavalry_reduce_value']+
        df['shaman_reduce_value']+
        df['wound_infantry_reduce_value']+
        df['wound_cavalry_reduce_value']+
        df['wound_shaman_reduce_value']+
        df['general_acceleration_reduce_value']+
        df['building_acceleration_reduce_value']+
        df['reaserch_acceleration_reduce_value']+
        df['training_acceleration_reduce_value']+
        df['treatment_acceleration_reduce_value'],
        df['wood_add_value']+
        df['stone_add_value']+
        df['ivory_add_value']+
        df['meat_add_value']+
        df['magic_add_value']+
        df['infantry_add_value']+
        df['cavalry_add_value']+
        df['shaman_add_value']+
        df['wound_infantry_add_value']+
        df['wound_cavalry_add_value']+
        df['wound_shaman_add_value']+
        df['general_acceleration_add_value']+
        df['building_acceleration_add_value']+
        df['reaserch_acceleration_add_value']+
        df['training_acceleration_add_value']+
        df['treatment_acceleraion_add_value']
    ,color='black')
#plt.yscale('log')
#plt.xscale('log')

plt.show()


# In[70]:


#资源使用率与付费
plt.scatter(
    (
        (df['wood_reduce_value']+
        df['stone_reduce_value']+
        df['ivory_reduce_value']+
        df['meat_reduce_value']+
        df['magic_reduce_value']+
        df['infantry_reduce_value']+
        df['cavalry_reduce_value']+
        df['shaman_reduce_value']+
        df['wound_infantry_reduce_value']+
        df['wound_cavalry_reduce_value']+
        df['wound_shaman_reduce_value']+
        df['general_acceleration_reduce_value']+
        df['building_acceleration_reduce_value']+
        df['reaserch_acceleration_reduce_value']+
        df['training_acceleration_reduce_value']+
        df['treatment_acceleration_reduce_value'])
          /
        (df['wood_add_value']+
        df['stone_add_value']+
        df['ivory_add_value']+
        df['meat_add_value']+
        df['magic_add_value']+
        df['infantry_add_value']+
        df['cavalry_add_value']+
        df['shaman_add_value']+
        df['wound_infantry_add_value']+
        df['wound_cavalry_add_value']+
        df['wound_shaman_add_value']+
        df['general_acceleration_add_value']+
        df['building_acceleration_add_value']+
        df['reaserch_acceleration_add_value']+
        df['training_acceleration_add_value']+
        df['treatment_acceleraion_add_value'])
     )
    , df['pay_price'],  color='black')
#plt.yscale('log')
#plt.xscale('log')

plt.show()


# In[55]:


#付费次数与付费
plt.scatter(df['pay_count']/df['pvp_battle_count'], df['pay_price'],  color='black')
#plt.yscale('log')
#plt.xscale('log')

plt.show()


# In[108]:


#在线时长与资源使用率
plt.scatter(df['avg_online_minutes'], (df['bd_training_hut_level']+
    df['bd_healing_lodge_level']+
    df['bd_stronghold_level']+
    df['bd_outpost_portal_level']+
    df['bd_barrack_level']+
    df['bd_healing_spring_level']+
    df['bd_dolmen_level']+
    df['bd_guest_cavern_level']+
    df['bd_warehouse_level']+
    df['bd_watchtower_level']+
    df['bd_magic_coin_tree_level']+
    df['bd_hall_of_war_level']+
    df['bd_market_level']+
    df['bd_hero_gacha_level']+
    df['bd_hero_strengthen_level']+
    df['bd_hero_pve_level']+
    df['sr_scout_level']+
    df['sr_training_speed_level']+
    df['sr_infantry_tier_2_level']+
    df['sr_cavalry_tier_2_level']+
    df['sr_shaman_tier_2_level']+
    df['sr_infantry_atk_level']+
    df['sr_cavalry_atk_level']+
    df['sr_shaman_atk_level']+
    df['sr_infantry_tier_3_level']+
    df['sr_cavalry_tier_3_level']+
    df['sr_shaman_tier_3_level']+
    df['sr_troop_defense_level']+
    df['sr_infantry_def_level']+
    df['sr_cavalry_def_level']+
    df['sr_shaman_def_level']+
    df['sr_infantry_hp_level']+
    df['sr_cavalry_hp_level']+
    df['sr_shaman_hp_level']+
    df['sr_infantry_tier_4_level']+
    df['sr_cavalry_tier_4_level']+
    df['sr_shaman_tier_4_level']+
    df['sr_troop_attack_level']+
    df['sr_construction_speed_level']+
    df['sr_hide_storage_level']+
    df['sr_troop_consumption_level']+
    df['sr_rss_a_prod_levell']+
    df['sr_rss_b_prod_level']+
    df['sr_rss_c_prod_level']+
    df['sr_rss_d_prod_level']+
    df['sr_rss_a_gather_level']+
    df['sr_rss_b_gather_level']+
    df['sr_rss_c_gather_level']+
    df['sr_rss_d_gather_level']+
    df['sr_troop_load_level']+
    df['sr_rss_e_gather_level']+
    df['sr_rss_e_prod_level']+
    df['sr_outpost_durability_level']+
    df['sr_outpost_tier_2_level']+
    df['sr_healing_space_level']+
    df['sr_gathering_hunter_buff_level']+
    df['sr_healing_speed_level']+
    df['sr_outpost_tier_3_level']+
    df['sr_alliance_march_speed_level']+
    df['sr_pvp_march_speed_level']+
    df['sr_gathering_march_speed_level']+
    df['sr_outpost_tier_4_level']+
    df['sr_guest_troop_capacity_level']+
    df['sr_march_size_level']+
    df['sr_rss_help_bonus_level']),  color='black')
#plt.yscale('log')
#plt.xscale('log')
plt.title('avg_online_minutes & levels')
plt.show()


#在线时长与资源使用率
plt.scatter((df['wood_reduce_value']+
        df['stone_reduce_value']+
        df['ivory_reduce_value']+
        df['meat_reduce_value']+
        df['magic_reduce_value']+
        df['infantry_reduce_value']+
        df['cavalry_reduce_value']+
        df['shaman_reduce_value']+
        df['wound_infantry_reduce_value']+
        df['wound_cavalry_reduce_value']+
        df['wound_shaman_reduce_value']+
        df['general_acceleration_reduce_value']+
        df['building_acceleration_reduce_value']+
        df['reaserch_acceleration_reduce_value']+
        df['training_acceleration_reduce_value']+
        df['treatment_acceleration_reduce_value'])/
        (df['wood_add_value']+
        df['stone_add_value']+
        df['ivory_add_value']+
        df['meat_add_value']+
        df['magic_add_value']+
        df['infantry_add_value']+
        df['cavalry_add_value']+
        df['shaman_add_value']+
        df['wound_infantry_add_value']+
        df['wound_cavalry_add_value']+
        df['wound_shaman_add_value']+
        df['general_acceleration_add_value']+
        df['building_acceleration_add_value']+
        df['reaserch_acceleration_add_value']+
        df['training_acceleration_add_value']+
        df['treatment_acceleraion_add_value']),  color='black')
#plt.yscale('log')
#plt.xscale('log')
plt.title('reduce & add')
plt.show()


# In[1]:


import pandas as pd


# ### user_id在train和test中有没有重复？

# In[5]:


# 读取数据
data = pd.read_csv("tap_fun_train.csv", parse_dates=True)
data_test = pd.read_csv("tap_fun_test.csv", parse_dates=True)

# 提取user_id列，并做合并处理
data_id = pd.DataFrame(data['user_id'],columns=['user_id'])
data_test_id = pd.DataFrame(data_test['user_id'],columns=['user_id'])
pd.merge(data_id, data_test_id, on = 'user_id')


# In[ ]:


## 训练集data清理及分类


# In[ ]:


data


# In[4]:


# 读取数据
data = pd.read_csv("tap_fun_train.csv", parse_dates=True)
#data_test = pd.read_csv("tap_fun_test.csv", parse_dates=True)

from pyecharts import Bar, Overlap

line3 = Line()
line3.add("注册玩家数量", data_day_count.index, data_day_count['register_time_day'], mark_line=["average"], mark_point=["max", "min"])

bar = Bar()
bar.add("45天内付费玩家比例", data_day_count.index, data_day_count['pay_percent'], yaxis_max=0.1, mark_point=["max", "min"])


overlap = Overlap()
overlap.add(line3)

overlap.add(bar, yaxis_index=1, is_add_yaxis=True)
overlap.render()

overlap

data_pay_45 = copy.copy(data[data['prediction_pay_price']!=0])

data_pay_45['prediction_pay_price']

import pandas as pd


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


# In[9]:


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

data['avg_online_minutes'].describe()
data_on=data[data['avg_online_minutes']!=0]
data=data.sort_values(by="avg_online_minutes",ascending= True)


plt.scatter(range(1,2245153),data_on['avg_online_minutes'])
#plt.xlim(-6,6)
plt.plot

plt.yscale('log')
#plt.xscale('log')


#付费玩家与非付费玩家df
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

from pyecharts import Line, Grid

line1 = Line("玩家数量统计-月")
line1.add("玩家数量", data_test_month_df.index, data_test_month_df['register_time_month'], mark_line=["average"], mark_point=["max", "min"])

line2 = Line("玩家数量统计-日",title_top="50%")
line2.add("玩家数量", data_test_day_df.index, data_test_day_df['register_time_day'], mark_line=["average"], mark_point=["max", "min"])

grid = Grid(width = 1000, height = 1000)
grid.add(line1, grid_bottom="60%")
grid.add(line2, grid_top="60%")
grid.render()

grid

data_test_pay_7 = copy.copy(data_test[data_test['pay_price']>0])
print(data_test_pay_7.shape)   # (19549, 110)
print(data_test_pay_7.shape[0]/data_test.shape[0])  # 0.023583300962440917

data_test_pay_7_day_df = pd.DataFrame(data_test_pay_7['register_time_day'].value_counts()).sort_index()
# print(data_pay_7_day_df)
data_test_pay_7_day_df.rename(columns={'register_time_day':'pay_register_time_day'}, inplace = True)
data_test_day_count = pd.concat([data_test_pay_7_day_df, data_test_day_df], axis=1)
# print(data_day_count)
data_test_day_count['pay_percent'] = data_test_day_count['pay_register_time_day']/data_test_day_count['register_time_day']
# print(data_day_count)

# ----------------------------- 画图
from pyecharts import Overlap

line3 = Line()
line3.add("注册玩家数量", data_test_day_count.index, data_test_day_count['register_time_day'], mark_line=["average"], mark_point=["max", "min"])

line4 = Line()
line4.add("7天内付费玩家数量", data_test_day_count.index, data_test_day_count['pay_register_time_day'], mark_line=["average"], 
          mark_point=["max", "min"], yaxis_max=3000)

overlap = Overlap()
# 默认不新增 x y 轴，并且 x y 轴的索引都为 0
overlap.add(line3)
# 新增一个 y 轴，此时 y 轴的数量为 2，第二个 y 轴的索引为 1（索引从 0 开始），所以设置 yaxis_index = 1
# 由于使用的是同一个 x 轴，所以 x 轴部分不用做出改变
overlap.add(line4, yaxis_index=1, is_add_yaxis=True)
overlap.render()

overlap


# In[27]:


from pyecharts import Bar, Overlap

line3 = Line()
line3.add("注册玩家数量", data_test_day_count.index, data_test_day_count['register_time_day'], mark_line=["average"], mark_point=["max", "min"])

bar = Bar()
bar.add("7天内付费玩家比例", data_test_day_count.index, data_test_day_count['pay_percent'], yaxis_max=0.1)


overlap = Overlap()
# 默认不新增 x y 轴，并且 x y 轴的索引都为 0
overlap.add(line3)
# 新增一个 y 轴，此时 y 轴的数量为 2，第二个 y 轴的索引为 1（索引从 0 开始），所以设置 yaxis_index = 1
# 由于使用的是同一个 x 轴，所以 x 轴部分不用做出改变
overlap.add(bar, yaxis_index=1, is_add_yaxis=True)
overlap.render()

overlap

data_test_pay_7 = copy.copy(data_test[data_test['pay_price']!=0])
print(data_test_pay_7['pay_price'].describe())
print('前7天合共付费：',data_test_pay_7['pay_price'].sum())  


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
# non-paid
sum(data_non_pay['wood_reduce_value'])/sum(data_non_pay['wood_add_value'])
sum(data_non_pay['stone_reduce_value'])/sum(data_non_pay['stone_add_value'])
sum(data_non_pay['ivory_reduce_value'])/sum(data_non_pay['ivory_add_value'])
sum(data_non_pay['meat_reduce_value'])/sum(data_non_pay['meat_add_value'])
sum(data_non_pay['magic_reduce_value'])/sum(data_non_pay['magic_add_value'])

#paid
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
# 付费玩家萨满损失率

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
# non-付费玩家萨满损失率 0.3825594082338661


# In[148]:


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

# non0-paid
print(
    sum(data_non_pay['general_acceleration_reduce_value'])/sum(data_non_pay['general_acceleration_add_value']),
    sum(data_non_pay['building_acceleration_reduce_value'])/sum(data_non_pay['building_acceleration_add_value']),
    sum(data_non_pay['reaserch_acceleration_reduce_value'])/sum(data_non_pay['reaserch_acceleration_add_value']),
    sum(data_non_pay['training_acceleration_reduce_value'])/sum(data_non_pay['training_acceleration_add_value']),
    sum(data_non_pay['treatment_acceleration_reduce_value'])/sum(data_non_pay['treatment_acceleraion_add_value'])
)


(sum(data_non_pay['general_acceleration_reduce_value'])+sum(data_non_pay['building_acceleration_reduce_value'])+sum(data_non_pay['reaserch_acceleration_reduce_value'])+sum(data_non_pay['training_acceleration_reduce_value']) +sum(data_pay['treatment_acceleration_reduce_value']))/(sum(data_non_pay['general_acceleration_add_value']) +sum(data_non_pay['building_acceleration_add_value'])+sum(data_non_pay['reaserch_acceleration_add_value']) +sum(data_non_pay['training_acceleration_add_value']) +sum(data_non_pay ['treatment_acceleraion_add_value']))

# 建筑等级
#paid
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

#non-paid
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

