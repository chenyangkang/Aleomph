#!/usr/bin/env /home/chenyangkang/city_and_pheno/miniconda3/envs/city/bin/python
# %% [markdown]
# # 1\. Load dependency

# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import arviz as az
import pymc as pm
import pickle
import seaborn as sns
from tqdm import tqdm
import time
from warnings import filterwarnings
import contextily as cx
import pickle

import math
from scipy.special import expit
    
import pymc.sampling.jax as pmjax

from fastprogress.fastprogress import master_bar, progress_bar
from fastprogress.fastprogress import force_console_behavior
master_bar, progress_bar = force_console_behavior()


filterwarnings('ignore')
pd.set_option('display.max_row', 100)

# %%
#####
wintering_prefix = '/data/wintering_model_level_3_May12'
breeding_prefix = '/data/breeding_model_level_3_May12'
input_final_df_prefix='/data/final_df_model_level_3_May12'
SAMPLE_SIZE=1000
SAMPLE_CHAINS=2
SAMPLE_CORES=2
TUNES=1000
 

# %% [markdown]
# # 2\. Load data

# %%
data=pd.read_csv('./8531.csv')
data.shape

# %%
plt.scatter(data.longitude, data.latitude, s=2)
# ax =plt.gca()
# cx.add_basemap(ax, crs='EPSG:4326')
plt.show()

# %%
data.latitude.mean(), data.longitude.mean()

# %%
data[data.urban_areas>0].shape, data[data.urban_areas==0].shape


# %%
pm.__version__

# %% [markdown]
# # 3\. Load taxonomy data and filter species list

# %%
#### read taxonomy data
ebird = pd.read_csv('./Bird_Functional_Traits/ebird_taxonomy/eBird_Taxonomy_v2021.csv')
avonet_raw = pd.read_csv('./Bird_Functional_Traits/AVONET/AVONET_Raw_Data.csv')
migration = pd.read_csv('./Bird_Functional_Traits/AVONET/AVONET1_BirdLife.csv')
taxon = ebird[['PRIMARY_COM_NAME','SCI_NAME','ORDER1','FAMILY']].drop_duplicates().rename(columns={'SCI_NAME':'Species2_eBird'}).merge(
    avonet_raw[['Species1_BirdLife','Species2_eBird']].drop_duplicates(),
    on='Species2_eBird', how='left'
    ).merge(
    migration[['Species1','Migration']].rename(columns={'Species1':'Species1_BirdLife'}),
    on='Species1_BirdLife', how='left'
    )
taxon = taxon[taxon['Migration']>1]


# %%
sp_list = list(data.columns)[list(data.columns).index('entropy')+1:]
sp_list = set(taxon.PRIMARY_COM_NAME.dropna().unique()) & set(sp_list)
sp_list = list(sp_list)
len(sp_list)

# %%
data[sp_list] = np.where(data[sp_list].fillna(0)>0,1,0)


# %%
### sorted sp list
sp_list = [i[0] for i in sorted(data[sp_list].sum().items(), key=lambda x:-x[1])]


# %% [markdown]
# # 4\. Filter input data by quality, demography, and migration type

# %%
def classify_mig_type(sub):
    
    spring_range = list(np.arange(60,152))
    summer_range = list(np.arange(152,245))
    fall_range = list(np.arange(245,337))
    winter_range = list(np.arange(1,60)) + list(np.arange(337,366))
    
    spring_occ = sub[sub.DOY.isin(spring_range)]['occ'].mean()
    summer_occ = sub[sub.DOY.isin(summer_range)]['occ'].mean()
    fall_occ = sub[sub.DOY.isin(fall_range)]['occ'].mean()
    winter_occ = sub[sub.DOY.isin(winter_range)]['occ'].mean()
    
    
    ###
    flag_list = []
    
    ### passgenger?
    if (spring_occ > winter_occ*2) & (fall_occ > winter_occ*2) & \
            (spring_occ > summer_occ*2) & (fall_occ > summer_occ*2):
        flag_list.append('passenger')
    
    if (winter_occ > summer_occ*2) & (winter_occ > spring_occ) & (winter_occ > fall_occ):
        flag_list.append('wintering')
    
    if (summer_occ > winter_occ*2) & (summer_occ > spring_occ) & (summer_occ > fall_occ):
        flag_list.append('breeding')
    
    return flag_list
        

# %%
## calculate cropland
data['cropland'] = data[['cropland_rainfed','cropland_irrigated_post_flooding','mosaic_cropland']].fillna(0).sum(axis=1).values



# %%
def assign_spatial_temporal_grid(sub):
    '''
        Spatial-temporal-urban-cropland subsampling. Five dimention. Booooom!
    '''
    lon_array = np.arange(sub.longitude.min(), sub.longitude.max(), 0.05)
    lat_array =  np.arange(sub.latitude.min(), sub.latitude.max(), 0.05)
    doy_array =  np.arange(sub.DOY.min(), sub.DOY.max(), 1)
    sub['lon_lat_doy_urban_crop_occ_grid'] = [str(i)+'_'+str(j)+'_'+str(k)+'_'+str(l)+'_'+str(m)+'_'+str(n) for i,j,k,l,m,n in zip(np.digitize(sub.longitude, lon_array),
                            np.digitize(sub.latitude, lat_array),
                            np.digitize(sub.DOY, doy_array),
                            np.where(sub.cropland>0,1,0),
                               np.where(sub.urban_areas>0,1,0),
                               np.where(sub.occ>0,1,0)
                               )]
    
    return sub


def down_sample(new_sub, samples_each_class=500):
    '''
        If the sample size is still too large -- subsample
    '''
    
    spring_range = list(np.arange(60,152)) + list(range(40,60)) + list(range(152,172))
    summer_range = list(np.arange(152,245)) 
    fall_range = list(np.arange(245,337)) + list(range(225,245)) + list(range(337,357))
    winter_range = list(np.arange(1,60)) + list(np.arange(337,366))
    
    occ_new_sub = new_sub[new_sub.occ==0]
    
    def get_sample_by_seasons(df_, samples_each_class):
        spring_df = df_[df_.DOY.isin(spring_range)]
        fall_df = df_[df_.DOY.isin(fall_range)]
        non_migrating_df = df_[(~df_.DOY.isin(spring_df)) & (~df_.DOY.isin(fall_range))]
        
        sample_size = samples_each_class//3
        
        def get_sample(df,sample_size):
            if len(df)>sample_size:
                df = df.sample(sample_size, replace=False) ### if data is abundant, first sample expert's checklist
            else: #.iloc[:samples_each_class,:].sort_values(by='obsvr_species_count', ascending=False)
                pass #df = df.sample(sample_size, replace=True)
            return df
        
        new_df_ = pd.concat(
            [get_sample(spring_df, sample_size),
            get_sample(fall_df, sample_size),
            get_sample(non_migrating_df, sample_size)],
            axis=0
        ).reset_index(drop=True)
        
        return new_df_
    
    occ_new_sub = get_sample_by_seasons(new_sub[new_sub.occ==0], samples_each_class)
    non_occ_new_sub = get_sample_by_seasons(new_sub[new_sub.occ>0], samples_each_class)
    return pd.concat([occ_new_sub, non_occ_new_sub], axis=0).reset_index(drop=True)




# %%
#### pre-process data: filter species, subsampling
final_list = []
for sp in tqdm(sp_list):
    sp_order = taxon[taxon['PRIMARY_COM_NAME']==sp]['ORDER1'].values[0]
    sub = data[['DOY','year','obsvr_species_count','longitude','latitude','urban_areas','cropland','duration_minutes',sp]]
    sub = sub[(sub.obsvr_species_count>sorted(sub.obsvr_species_count)[int(len(sub)*0.05)]) &\
        (sub.obsvr_species_count<sorted(sub.obsvr_species_count)[int(len(sub)*0.95)])] #### remove outlier birder with extreme bird species count
    sub['Order'] = sp_order
    sub[sp] = sub[sp].fillna(0)
    sub=sub.dropna().rename(columns={sp:'occ'})
    sub['occ'] = np.where(sub['occ']>0,1,0)
    sub['sp'] = sp
    ### filtering
#     mean_occ = sub[['DOY','occ']].groupby('DOY').mean()
#     if (mean_occ.std()<0.01).values[0]:
#         continue
    ### only take summer breeding indiv
    flag_list = classify_mig_type(sub)
    if len(flag_list)>1 or len(flag_list)==0:
        # print('Error: ',flag_list)
        continue
    else:
        flag=flag_list[0]
        
    #### filtering for each year
    sp_df_list = []
    for year in sorted(sub.year.unique()):
        ### filtering
        subsub=sub[sub.year==year]
        if subsub[subsub.occ>0].shape[0]<250:
            continue

        ### resample    
        ### subsample
        subsub = assign_spatial_temporal_grid(subsub)
        subsub = subsub.sample(frac=1, replace=False).groupby('lon_lat_doy_urban_crop_occ_grid').first()
        if len(subsub[(subsub.urban_areas>0) & (subsub.occ>0)])<30:
            continue
        if len(subsub[(subsub.cropland>0) & (subsub.occ>0)])<30:
            continue
            
        if len(subsub)>500: #.sort_values(by='obsvr_species_count',ascending=False)
            subsub = down_sample(sub, samples_each_class=150) 
            
        subsub['migration_type'] = flag
        sp_df_list.append(subsub)
    
    if len(sp_df_list)==0:
        continue
    
    print(flag,end='..')
    print(f'{len(sp_df_list)} years samples',end='..')
    final_list.append(pd.concat(sp_df_list, axis=0).reset_index(drop=True))
    
    
print('total species: ',len(final_list))
final_df = pd.concat(final_list, axis=0).reset_index(drop=True)






# %%
def reindexing(df):
    df['record_index'] = list(range(len(df)))
    df['year_index'] = pd.factorize(df.year)[0]
    df['order_index'] = pd.factorize(df.Order)[0]
    df['sp_index'] = pd.factorize(df.sp)[0]
    df['duration_minutes'] = (df['duration_minutes'] - df['duration_minutes'].mean())/df['duration_minutes'].std()
    # df['urban_areas'] = (df['urban_areas'] - df['urban_areas'].mean())/df['urban_areas'].std()
    # df['cropland'] = (df['cropland'] - df['cropland'].mean())/df['cropland'].std()
    return df

# %%
### save the final input file
final_df.to_csv(f'{input_final_df_prefix}.csv',index=False)

# %%
### split by patter
passenger_df = reindexing(
    final_df[final_df.migration_type=='passenger']
    )
breeding_df = reindexing(
    final_df[final_df.migration_type=='breeding']
)
wintering_df = reindexing(
    final_df[final_df.migration_type=='wintering']
)
print(passenger_df.shape,breeding_df.shape, wintering_df.shape)


### save the final input file
wintering_df.to_csv(f'{wintering_prefix}.csv',index=False)
breeding_df.to_csv(f'{breeding_prefix}.csv',index=False)


# %%
wintering_df.sp.unique()

# %%
# plt.plot(passenger_df[['DOY','occ']].groupby('DOY').mean().rolling(10).mean())
plt.plot(passenger_df[passenger_df.urban_areas>0][['DOY','occ']].groupby('DOY').mean().rolling(10).mean(), label='urban')
plt.plot(passenger_df[passenger_df.urban_areas==0][['DOY','occ']].groupby('DOY').mean().rolling(10).mean(),label='nature')
plt.legend()
plt.show()



# %%

plt.plot(wintering_df[wintering_df.urban_areas>0][['DOY','occ']].groupby('DOY').mean().rolling(10).mean(), label='urban')
plt.plot(wintering_df[wintering_df.urban_areas==0][['DOY','occ']].groupby('DOY').mean().rolling(10).mean(),label='nature')
plt.legend()
plt.show()


# %%

plt.plot(breeding_df[breeding_df.urban_areas>0][['DOY','occ']].groupby('DOY').mean().rolling(10).mean(), label='urban')
plt.plot(breeding_df[breeding_df.urban_areas==0][['DOY','occ']].groupby('DOY').mean().rolling(10).mean(),label='nature')
plt.legend()
plt.show()


# %%


# %% [markdown]
# # 5\. Spring departure model

# %% [markdown]
# ## 5\.0 Train test split

# %%
from sklearn.model_selection import train_test_split
X_train_list = []
X_test_list = []
y_train_list = []
y_test_list = []

for sp in wintering_df.sp.unique():
    sub = wintering_df[wintering_df.sp==sp]
    
    X_train, X_test, y_train, y_test = train_test_split(sub[[i for i in sub.columns if not i=='occ']],
                                                        sub['occ'],test_size=0.2)
    X_train_list.append(X_train)
    X_test_list.append(X_test)
    y_train_list.append(y_train)
    y_test_list.append(y_test)
    
    
X_train = pd.concat(X_train_list, axis=0)
X_test = pd.concat(X_test_list, axis=0)
y_train = pd.concat(y_train_list, axis=0)
y_test = pd.concat(y_test_list, axis=0)


# %% [markdown]
# ## 5\.1 Define model

# %%
### passenger model


# ### running

wintering_model = pm.Model()

with wintering_model:
    
    def cumulative_normal(x, mu, sigma, s=pm.math.sqrt(2)):
        return 0.5 + 0.5 * pm.math.erf((x-mu)/(sigma*s))
    def neg_inverse_cumulative_normal(x, mu, sigma, s=pm.math.sqrt(2)):
        return -0.5 - 0.5 * pm.math.erf((x-mu)/(sigma*s))

    wintering_model.add_coord("record_index", sorted(wintering_df['record_index'].unique()))
    wintering_model.add_coord("sp_index", sorted(wintering_df['sp_index'].unique()))
    wintering_model.add_coord("year_index", sorted(wintering_df['year_index'].unique()))
    wintering_model.add_coord("order_index", sorted(wintering_df['order_index'].unique()))
    
    
    DOY_obs = pm.Data('DOY_obs',X_train.DOY.values)
    occ_obs = pm.Data('occ_obs',y_train.values)
    urban_areas_obs = pm.Data('urban_areas_obs',X_train.urban_areas.values)
    cropland_obs = pm.Data('cropland_obs',X_train.cropland.values)
    duration_obs = pm.Data('duration_obs',X_train.duration_minutes.values)
    
    sp_index_obs = pm.Data('sp_index_obs', X_train.sp_index.values)
    year_index_obs = pm.Data('year_index_obs', X_train.year_index.values)
#     order_index_obs = pm.Data('order_index_obs', X_train.order_index.values)
    order_to_sp_obs = pm.Data('order_to_sp_obs', X_train[['sp_index','order_index']].drop_duplicates().sort_values(by='sp_index')['order_index'].values)
    
    ### prior
    ## spring departure
    spring_departure_dist_mu_overall = pm.TruncatedNormal('spring_departure_dist_mu_overall', mu=106, sigma=200, lower=20, upper=175)
    spring_departure_dist_mu = pm.Normal('spring_departure_dist_mu', mu=spring_departure_dist_mu_overall, sigma=30, dims=('sp_index','year_index'))
    spring_departure_dist_sigma = pm.HalfNormal('spring_departure_dist_sigma', sigma=30, dims=('sp_index','year_index'))
    
    ## fall arrival
    fall_arrival_dist_mu_overall = pm.TruncatedNormal('fall_arrival_dist_mu_overall', mu=291, sigma=200, lower=175, upper=356)
    fall_arrival_dist_mu = pm.Normal('fall_arrival_dist_mu', mu=fall_arrival_dist_mu_overall, sigma=30, dims=('sp_index','year_index'))
    fall_arrival_dist_sigma = pm.HalfNormal('fall_arrival_dist_sigma', sigma=30, dims=('sp_index','year_index'))
    
    ##### intercept
    beta = pm.Normal('beta', mu=0, sigma=50,dims=('sp_index','year_index'))
    
    #### urban effect   
    spring_departure_urban_effect_overall = pm.Normal('spring_departure_urban_effect_overall', mu=0, sigma=50)
    spring_departure_urban_effect_order_level = pm.Normal('spring_departure_urban_effect_order_level', mu=spring_departure_urban_effect_overall, sigma=50, dims=('order_index'))
    spring_departure_urban_effect = pm.Normal('spring_departure_urban_effect', mu=spring_departure_urban_effect_order_level[order_to_sp_obs], 
                                              sigma=50, dims=('sp_index'))
    fall_arrival_urban_effect_overall = pm.Normal('fall_arrival_urban_effect_overall', mu=0, sigma=50)
    fall_arrival_urban_effect_order_level = pm.Normal('fall_arrival_urban_effect_order_level', mu=fall_arrival_urban_effect_overall, sigma=50, dims=('order_index'))
    fall_arrival_urban_effect = pm.Normal('fall_arrival_urban_effect', mu=fall_arrival_urban_effect_order_level[order_to_sp_obs],
                                          sigma=50, dims=('sp_index'))
    
    #### cropland effect   
    spring_departure_cropland_effect_overall = pm.Normal('spring_departure_cropland_effect_overall', mu=0, sigma=50)
    spring_departure_cropland_effect_order_level = pm.Normal('spring_departure_cropland_effect_order_level', mu=spring_departure_cropland_effect_overall, sigma=50, dims=('order_index'))
    spring_departure_cropland_effect = pm.Normal('spring_departure_cropland_effect', mu=spring_departure_cropland_effect_order_level[order_to_sp_obs], 
                                                 sigma=50, dims=('sp_index'))
    fall_arrival_cropland_effect_overall = pm.Normal('fall_arrival_cropland_effect_overall', mu=0, sigma=50)
    fall_arrival_cropland_effect_order_level = pm.Normal('fall_arrival_cropland_effect_order_level', mu=fall_arrival_cropland_effect_overall, sigma=50, dims=('order_index'))
    fall_arrival_cropland_effect = pm.Normal('fall_arrival_cropland_effect', mu=fall_arrival_cropland_effect_order_level[order_to_sp_obs], 
                                             sigma=50, dims=('sp_index'))
    
    ### duration effect
    duration_effect_slope = pm.Normal('duration_effect_slope', mu=0, sigma=100, dims=('sp_index'))
    
    #### city effect modification on timing
    modified_spring_departure_dist = spring_departure_dist_mu[sp_index_obs, year_index_obs] + \
                                        spring_departure_urban_effect[sp_index_obs] * urban_areas_obs + \
                                            spring_departure_cropland_effect[sp_index_obs] * cropland_obs
    
    modified_fall_arrival_dist = fall_arrival_dist_mu[sp_index_obs, year_index_obs] + \
                                    fall_arrival_urban_effect[sp_index_obs] * urban_areas_obs + \
                                        fall_arrival_cropland_effect[sp_index_obs] * cropland_obs


    ### scaling factor
    spring_departure_cumulative_effect_factor = pm.Normal('spring_departure_cumulative_effect_factor',mu=0,sigma=100, dims=('sp_index','year_index'))
    fall_arrival_cumulative_effect_factor = pm.Normal('fall_arrival_cumulative_effect_factor',mu=0,sigma=100, dims=('sp_index','year_index'))
    
    ### variables times scalers
    duration_effect_on_obs_prob = duration_effect_slope[sp_index_obs] * duration_obs
    spring_departure_effect_on_obs_prob = (neg_inverse_cumulative_normal(x=DOY_obs, 
                                                                         mu=modified_spring_departure_dist, 
                                                                         sigma=spring_departure_dist_sigma[sp_index_obs, year_index_obs]) + 0.5) * spring_departure_cumulative_effect_factor[sp_index_obs, year_index_obs]
    fall_arrival_effect_on_obs_prob = (cumulative_normal(x=DOY_obs, 
                                                         mu=modified_fall_arrival_dist, 
                                                         sigma=fall_arrival_dist_sigma[sp_index_obs, year_index_obs]) - 0.5)* fall_arrival_cumulative_effect_factor[sp_index_obs, year_index_obs]

    #### linear term
    y0 = beta[sp_index_obs, year_index_obs] + \
            duration_effect_on_obs_prob + \
            spring_departure_effect_on_obs_prob + \
            fall_arrival_effect_on_obs_prob
            
    p_ = pm.math.sigmoid(y0)
    
    y_ = pm.Bernoulli("y_", p_, observed=occ_obs)
    

    

    

# %% [markdown]
# ## 5\.2 Sample chains

# %%


# %%


with wintering_model:
    idata = pmjax.sample_numpyro_nuts(SAMPLE_SIZE,chains=SAMPLE_CHAINS,cores=SAMPLE_CORES, tune=TUNES, progressbar=True)
    
    #  target_accept=0.99,
    #                   chains=2, cores=2, tune=3000, progressbar=True,
    #                     # trace = ['y_']
                      

# %%
with wintering_model:
    post_idata = pm.sample_posterior_predictive(idata)

# %% [markdown]
# ## 5\.3 Predict on test data

# %%
def pm_predict_winter_model(X_test):
    ### passenger model
    def math_cumulative_normal(x, mu, sigma, s=np.sqrt(2)):
        return 0.5 + 0.5 * math.erf((x-mu)/(sigma*s))
    def math_neg_inverse_cumulative_normal(x, mu, sigma, s=np.sqrt(2)):
        return -0.5 - 0.5 * math.erf((x-mu)/(sigma*s))

    new_DOY_obs = X_test['DOY'].values
    new_sp_index_obs = X_test['sp_index'].values
    new_year_index_obs = X_test['year_index'].values
    new_urban_areas_obs = X_test['urban_areas'].values
    new_cropland_obs = X_test['cropland'].values

    new_beta = np.concatenate(idata.posterior['beta'].values, axis=0).mean(axis=0)[new_sp_index_obs, new_year_index_obs]
    new_duration_effect_on_obs_prob = np.concatenate(idata.posterior['duration_effect_slope'].values, axis=0).mean(axis=0)[new_sp_index_obs]

    new_modified_spring_departure_dist = np.concatenate(idata.posterior['spring_departure_dist_mu'], axis=0).mean(axis=0)[new_sp_index_obs, new_year_index_obs] + \
                                            np.concatenate(idata.posterior['spring_departure_urban_effect'], axis=0).mean(axis=0)[new_sp_index_obs] * new_urban_areas_obs + \
                                                np.concatenate(idata.posterior['spring_departure_cropland_effect'], axis=0).mean(axis=0)[new_sp_index_obs] * new_cropland_obs
                                                
    new_spring_departure_effect_on_obs_prob =  (
        np.array([math_neg_inverse_cumulative_normal(x=x,mu=mu,sigma=sigma) for x,mu,sigma in zip(
        new_DOY_obs,
        new_modified_spring_departure_dist, 
        np.concatenate(idata.posterior['spring_departure_dist_sigma'], axis=0).mean(axis=0)[new_sp_index_obs, new_year_index_obs]
    )]) + 0.5) * np.concatenate(idata.posterior['spring_departure_cumulative_effect_factor'], axis=0).mean(axis=0)[new_sp_index_obs, new_year_index_obs]

    new_modified_fall_arrival_dist = np.concatenate(idata.posterior['fall_arrival_dist_mu'], axis=0).mean(axis=0)[new_sp_index_obs, new_year_index_obs] + \
                                        np.concatenate(idata.posterior['fall_arrival_urban_effect'], axis=0).mean(axis=0)[new_sp_index_obs] * new_urban_areas_obs + \
                                            np.concatenate(idata.posterior['fall_arrival_cropland_effect'], axis=0).mean(axis=0)[new_sp_index_obs] * new_cropland_obs
                                            
    new_fall_arrival_effect_on_obs_prob = (np.array([
        math_cumulative_normal(x=x,mu=mu,sigma=sigma) for x,mu,sigma in zip(
            new_DOY_obs,
            new_modified_fall_arrival_dist,
            np.concatenate(idata.posterior['fall_arrival_dist_sigma'], axis=0).mean(axis=0)[new_sp_index_obs, new_year_index_obs]
        )
    ]) - 0.5)* np.concatenate(idata.posterior['fall_arrival_cumulative_effect_factor'], axis=0).mean(axis=0)[new_sp_index_obs, new_year_index_obs]

    new_y0 = new_beta + \
            new_duration_effect_on_obs_prob + \
            new_spring_departure_effect_on_obs_prob + \
            new_fall_arrival_effect_on_obs_prob
            
    new_p_ = expit(new_y0)
    y_pred = np.where(new_p_>0.5,1,0)
    return y_pred





# %%
y_pred = pm_predict_winter_model(X_test)

# %%
post_bi = np.where(
    np.concatenate(post_idata.posterior_predictive['y_'].values, axis=0).mean(axis=0)>0.5,
    1,0
)


# %%
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score

print(
    roc_auc_score(y_train, post_bi), precision_score(y_train, post_bi), recall_score(y_train, post_bi), f1_score(y_train, post_bi)
)

print(
    roc_auc_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred), f1_score(y_test, y_pred)
)


# %% [markdown]
# ## 5\.4 Evaluate and save data

# %%
###
ess_ = az.ess(idata)
rhat_ = az.rhat(idata)
bfmi_ = az.bfmi(idata)
    
### save data
with open(f'{wintering_prefix}.pkl','wb') as f:
    pickle.dump({'idata':idata,'post_data':post_idata, ### trace and posterior predictives
                 'ess':ess_,'rhat':rhat_,'bmfi':bfmi_, ### evaluations
                 'train_roc_auc':roc_auc_score(y_train, post_bi),
                 'train_precision':precision_score(y_train, post_bi), 
                 'train_recall':recall_score(y_train, post_bi), 
                 'train_f1':f1_score(y_train, post_bi),
                 
                 'test_roc_auc':roc_auc_score(y_test, y_pred),
                 'test_precision':precision_score(y_test, y_pred), 
                 'test_recall':recall_score(y_test, y_pred), 
                 'test_f1':f1_score(y_test, y_pred)
                 }, f)


# %%


# %% [markdown]
# # 6\. Fall departure model

# %% [markdown]
# ## 6\.0 Train test split

# %%
from sklearn.model_selection import train_test_split
X_train_list = []
X_test_list = []
y_train_list = []
y_test_list = []

for sp in breeding_df.sp.unique():
    sub = breeding_df[breeding_df.sp==sp]
    
    X_train, X_test, y_train, y_test = train_test_split(sub[[i for i in sub.columns if not i=='occ']],
                                                        sub['occ'],test_size=0.2)
    X_train_list.append(X_train)
    X_test_list.append(X_test)
    y_train_list.append(y_train)
    y_test_list.append(y_test)
    
    
X_train = pd.concat(X_train_list, axis=0)
X_test = pd.concat(X_test_list, axis=0)
y_train = pd.concat(y_train_list, axis=0)
y_test = pd.concat(y_test_list, axis=0)


# %% [markdown]
# ## 6\.1 Define model

# %%
### passenger model
def cumulative_normal(x, mu, sigma, s=pm.math.sqrt(2)):
    return 0.5 + 0.5 * pm.math.erf((x-mu)/(sigma*s))
def neg_inverse_cumulative_normal(x, mu, sigma, s=pm.math.sqrt(2)):
    return -0.5 - 0.5 * pm.math.erf((x-mu)/(sigma*s))

# ### running

breeding_model = pm.Model()

with breeding_model:
    breeding_model.add_coord("record_index", sorted(breeding_df['record_index'].unique()))
    breeding_model.add_coord("sp_index", sorted(breeding_df['sp_index'].unique()))
    breeding_model.add_coord("year_index", sorted(breeding_df['year_index'].unique()))
    breeding_model.add_coord("order_index", sorted(breeding_df['order_index'].unique()))
    
    
    DOY_obs = pm.Data('DOY_obs',X_train.DOY.values)
    occ_obs = pm.Data('occ_obs',y_train.values)
    urban_areas_obs = pm.Data('urban_areas_obs',X_train.urban_areas.values)
    cropland_obs = pm.Data('cropland_obs',X_train.cropland.values)
    duration_obs = pm.Data('duration_obs',X_train.duration_minutes.values)
    
    sp_index_obs = pm.Data('sp_index_obs', X_train.sp_index.values)
    year_index_obs = pm.Data('year_index_obs', X_train.year_index.values)
    order_to_sp_obs = pm.Data('order_to_sp_obs', X_train[['sp_index','order_index']].drop_duplicates().sort_values(by='sp_index')['order_index'].values)
    
    ### prior
    ## spring arrival
    spring_arrival_dist_mu_overall = pm.TruncatedNormal('spring_arrival_dist_mu_overall', mu=106, sigma=20, lower=20, upper=175)
    spring_arrival_dist_mu = pm.Normal('spring_arrival_dist_mu', mu=spring_arrival_dist_mu_overall, sigma=30, dims=('sp_index','year_index'))
    spring_arrival_dist_sigma = pm.HalfNormal('spring_arrival_dist_sigma', sigma=30, dims=('sp_index','year_index'))
    # spring_arrival_dist = pm.Normal('spring_arrival_dist', mu=spring_arrival_dist_mu, sigma=spring_arrival_dist_sigma)
    
    ## fall departure
    fall_departure_dist_mu_overall = pm.TruncatedNormal('fall_departure_dist_mu_overall', mu=291, sigma=200, lower=175, upper=356)
    fall_departure_dist_mu = pm.Normal('fall_departure_dist_mu', mu=fall_departure_dist_mu_overall, sigma=30, dims=('sp_index','year_index'))
    fall_departure_dist_sigma = pm.HalfNormal('fall_departure_dist_sigma', sigma=30, dims=('sp_index','year_index'))
    # fall_departure_dist = pm.Normal('fall_departure_dist', mu=fall_departure_dist_mu, sigma=fall_departure_dist_sigma, dims=('sp_index'))
    
    ##### intercept
    beta = pm.Normal('beta', mu=0, sigma=50,dims=('sp_index','year_index'))
    
    #### urban effect   
    spring_arrival_urban_effect_overall = pm.Normal('spring_arrival_urban_effect_overall', mu=0, sigma=50)
    spring_arrival_urban_effect_order_level = pm.Normal('spring_arrival_urban_effect_order_level', mu=spring_arrival_urban_effect_overall, sigma=50, dims=('order_index'))
    spring_arrival_urban_effect = pm.Normal('spring_arrival_urban_effect', mu=spring_arrival_urban_effect_order_level[order_to_sp_obs], 
                                              sigma=50, dims=('sp_index'))
    fall_departure_urban_effect_overall = pm.Normal('fall_departure_urban_effect_overall', mu=0, sigma=50)
    fall_departure_urban_effect_order_level = pm.Normal('fall_departure_urban_effect_order_level', mu=fall_departure_urban_effect_overall, sigma=50, dims=('order_index'))
    fall_departure_urban_effect = pm.Normal('fall_departure_urban_effect', mu=fall_departure_urban_effect_order_level[order_to_sp_obs],
                                          sigma=50, dims=('sp_index'))
    
    #### cropland effect   
    spring_arrival_cropland_effect_overall = pm.Normal('spring_arrival_cropland_effect_overall', mu=0, sigma=50)
    spring_arrival_cropland_effect_order_level = pm.Normal('spring_arrival_cropland_effect_order_level', mu=spring_arrival_cropland_effect_overall, sigma=50, dims=('order_index'))
    spring_arrival_cropland_effect = pm.Normal('spring_arrival_cropland_effect', mu=spring_arrival_cropland_effect_order_level[order_to_sp_obs], 
                                                 sigma=50, dims=('sp_index'))
    fall_departure_cropland_effect_overall = pm.Normal('fall_departure_cropland_effect_overall', mu=0, sigma=50)
    fall_departure_cropland_effect_order_level = pm.Normal('fall_departure_cropland_effect_order_level', mu=fall_departure_cropland_effect_overall, sigma=50, dims=('order_index'))
    fall_departure_cropland_effect = pm.Normal('fall_departure_cropland_effect', mu=fall_departure_cropland_effect_order_level[order_to_sp_obs], 
                                             sigma=50, dims=('sp_index'))
    
    ### duration effect
    duration_effect_slope = pm.Normal('duration_effect_slope', sigma=100, dims=('sp_index'))
    
    #### city effect modification on timing
    modified_spring_arrival_dist = spring_arrival_dist_mu[sp_index_obs, year_index_obs] + \
                                        spring_arrival_urban_effect[sp_index_obs] * urban_areas_obs + \
                                            spring_arrival_cropland_effect[sp_index_obs] * cropland_obs
    
    modified_fall_departure_dist = fall_departure_dist_mu[sp_index_obs, year_index_obs] + \
                                    fall_departure_urban_effect[sp_index_obs] * urban_areas_obs + \
                                        fall_departure_cropland_effect[sp_index_obs] * cropland_obs


    ### scaling factor
    spring_arrival_cumulative_effect_factor = pm.Normal('spring_arrival_cumulative_effect_factor',mu=0,sigma=100, dims=('sp_index','year_index'))
    fall_departure_cumulative_effect_factor = pm.Normal('fall_departure_cumulative_effect_factor',mu=0,sigma=100, dims=('sp_index','year_index'))
    
    ### variables times scalers
    duration_effect_on_obs_prob = duration_effect_slope[sp_index_obs] * duration_obs
    spring_arrival_effect_on_obs_prob = (cumulative_normal(x=DOY_obs, 
                                                                         mu=modified_spring_arrival_dist, 
                                                                         sigma=spring_arrival_dist_sigma[sp_index_obs, year_index_obs]) - 0.5) * spring_arrival_cumulative_effect_factor[sp_index_obs, year_index_obs]
    fall_departure_effect_on_obs_prob = (neg_inverse_cumulative_normal(x=DOY_obs, 
                                                         mu=modified_fall_departure_dist, 
                                                         sigma=fall_departure_dist_sigma[sp_index_obs, year_index_obs]) + 0.5)* fall_departure_cumulative_effect_factor[sp_index_obs, year_index_obs]

    #### linear term
            
    y0 = beta[sp_index_obs, year_index_obs] + \
            duration_effect_on_obs_prob + \
            spring_arrival_effect_on_obs_prob + \
            fall_departure_effect_on_obs_prob
            
    p_ = pm.math.sigmoid(y0)
    
    y_ = pm.Bernoulli("y_", p_, observed=occ_obs)
    

    

    

# %% [markdown]
# ## 6\.2 Sample chains

# %%
with breeding_model:
    idata = pmjax.sample_numpyro_nuts(SAMPLE_SIZE,chains=SAMPLE_CHAINS,cores=SAMPLE_CORES, tune=TUNES, progressbar=True)
    
    #  target_accept=0.99,
    #                   chains=2, cores=2, tune=3000, progressbar=True,
    #                     # trace = ['y_']
                      

# %%
with breeding_model:
    post_idata = pm.sample_posterior_predictive(idata)

# %% [markdown]
# ## 6\.3 Predict on test data

# %%
def pm_predict_breeding_model(X_test):
    ### passenger model

    def math_cumulative_normal(x, mu, sigma, s=np.sqrt(2)):
        return 0.5 + 0.5 * math.erf((x-mu)/(sigma*s))
    def math_neg_inverse_cumulative_normal(x, mu, sigma, s=np.sqrt(2)):
        return -0.5 - 0.5 * math.erf((x-mu)/(sigma*s))

    new_DOY_obs = X_test['DOY'].values
    new_sp_index_obs = X_test['sp_index'].values
    new_year_index_obs = X_test['year_index'].values
    new_urban_areas_obs = X_test['urban_areas'].values
    new_cropland_obs = X_test['cropland'].values

    new_beta = np.concatenate(idata.posterior['beta'].values, axis=0).mean(axis=0)[new_sp_index_obs, new_year_index_obs]
    new_duration_effect_on_obs_prob = np.concatenate(idata.posterior['duration_effect_slope'].values, axis=0).mean(axis=0)[new_sp_index_obs]

    new_modified_spring_arrival_dist = np.concatenate(idata.posterior['spring_arrival_dist_mu'], axis=0).mean(axis=0)[new_sp_index_obs, new_year_index_obs] + \
                                            np.concatenate(idata.posterior['spring_arrival_urban_effect'], axis=0).mean(axis=0)[new_sp_index_obs] * new_urban_areas_obs + \
                                                np.concatenate(idata.posterior['spring_arrival_cropland_effect'], axis=0).mean(axis=0)[new_sp_index_obs] * new_cropland_obs
                                                
    new_spring_arrival_effect_on_obs_prob =  (
        np.array([math_cumulative_normal(x=x,mu=mu,sigma=sigma) for x,mu,sigma in zip(
        new_DOY_obs,
        new_modified_spring_arrival_dist, 
        np.concatenate(idata.posterior['spring_arrival_dist_sigma'], axis=0).mean(axis=0)[new_sp_index_obs, new_year_index_obs]
    )]) - 0.5) * np.concatenate(idata.posterior['spring_arrival_cumulative_effect_factor'], axis=0).mean(axis=0)[new_sp_index_obs, new_year_index_obs]

    new_modified_fall_departure_dist = np.concatenate(idata.posterior['fall_departure_dist_mu'], axis=0).mean(axis=0)[new_sp_index_obs, new_year_index_obs] + \
                                        np.concatenate(idata.posterior['fall_departure_urban_effect'], axis=0).mean(axis=0)[new_sp_index_obs] * new_urban_areas_obs + \
                                            np.concatenate(idata.posterior['fall_departure_cropland_effect'], axis=0).mean(axis=0)[new_sp_index_obs] * new_cropland_obs
                                            
    new_fall_departure_effect_on_obs_prob = (np.array([
        math_neg_inverse_cumulative_normal(x=x,mu=mu,sigma=sigma) for x,mu,sigma in zip(
            new_DOY_obs,
            new_modified_fall_departure_dist,
            np.concatenate(idata.posterior['fall_departure_dist_sigma'], axis=0).mean(axis=0)[new_sp_index_obs, new_year_index_obs]
        )
    ]) + 0.5)* np.concatenate(idata.posterior['fall_departure_cumulative_effect_factor'], axis=0).mean(axis=0)[new_sp_index_obs, new_year_index_obs]

    new_y0 = new_beta + \
            new_duration_effect_on_obs_prob + \
            new_spring_arrival_effect_on_obs_prob + \
            new_fall_departure_effect_on_obs_prob
            
    new_p_ = expit(new_y0)
    y_pred = np.where(new_p_>0.5,1,0)
    return y_pred





# %%
y_pred = pm_predict_breeding_model(X_test)

# %%
post_bi = np.where(
    np.concatenate(post_idata.posterior_predictive['y_'].values, axis=0).mean(axis=0)>0.5,
    1,0
)


# %%
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score

print(
    roc_auc_score(y_train, post_bi), precision_score(y_train, post_bi), recall_score(y_train, post_bi), f1_score(y_train, post_bi)
)

print(
    roc_auc_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred), f1_score(y_test, y_pred)
)


# %% [markdown]
# ## 6\.4 Evaluate and save data

# %%
###
ess_ = az.ess(idata)
rhat_ = az.rhat(idata)
bfmi_ = az.bfmi(idata)
    
### save data
with open(f'{breeding_prefix}.pkl','wb') as f:
    pickle.dump({'idata':idata,'post_data':post_idata, ### trace and posterior predictives
                 'ess':ess_,'rhat':rhat_,'bmfi':bfmi_, ### evaluations
                 'train_roc_auc':roc_auc_score(y_train, post_bi),
                 'train_precision':precision_score(y_train, post_bi), 
                 'train_recall':recall_score(y_train, post_bi), 
                 'train_f1':f1_score(y_train, post_bi),
                 
                 'test_roc_auc':roc_auc_score(y_test, y_pred),
                 'test_precision':precision_score(y_test, y_pred), 
                 'test_recall':recall_score(y_test, y_pred), 
                 'test_f1':f1_score(y_test, y_pred)
                 }, f)


# %% [markdown]
# 

# %% [markdown]
# 

# %%


# %%


# %%


# %%


# %%


# %%



