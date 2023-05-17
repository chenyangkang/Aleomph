#!/usr/bin/env /home/chenyangkang/city_and_pheno/miniconda3/envs/city/bin/python
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
filterwarnings('ignore')
pd.set_option('display.max_row', 100)

# %%
#####
wintering_prefix = '/data/wintering_model_level_3_May17'
breeding_prefix = '/data/breeding_model_level_3_May17'
input_final_df_prefix='/data/final_df_model_level_3_May17'
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
        Spatial-temporal-urban-cropland subsampling. Five dimension. Booooom!
    '''
    lon_array = np.arange(sub.longitude.min(), sub.longitude.max(), 0.01)
    lat_array =  np.arange(sub.latitude.min(), sub.latitude.max(), 0.01)
    doy_array =  np.arange(sub.DOY.min(), sub.DOY.max(), 1)
    sub['lon_lat_doy_urban_crop_grid'] = [str(i)+'_'+str(j)+'_'+str(k) for i,j,k in zip(np.digitize(sub.longitude, lon_array),
                            np.digitize(sub.latitude, lat_array),
                            np.digitize(sub.DOY, doy_array)
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
def get_ml_score(X, y):
    ## can i use DOY, urban, cropland, duration to predict occ?
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score, precision_score, accuracy_score,make_scorer
    
    model = RandomForestClassifier(random_state=42,class_weight='balanced')
    scores = cross_val_score(model, X, y, cv=5, scoring=make_scorer(roc_auc_score))
    return np.nanmean(scores)

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
    ### only take summer breeding indiv
    flag_list = classify_mig_type(sub)
    if len(flag_list)>1 or len(flag_list)==0:
        continue
    else:
        flag=flag_list[0]
        
    #### filtering for each year
    sp_df_list = []
    for year in sorted(sub.year.unique()):
        ### filtering
        subsub=sub[sub.year==year]

        ### resample    
        ### subsample
        subsub = assign_spatial_temporal_grid(subsub)
        subsub = subsub.sample(frac=1, replace=False).groupby('lon_lat_doy_urban_crop_grid').first()
        
        if subsub[subsub.occ>0].shape[0]<100: ### after subsampling, if less than 100 observations, skip
            continue
        if len(subsub[(subsub.urban_areas>0.05) & (subsub.occ>0)])<30:
            continue
        if len(subsub[(subsub.cropland>0.05) & (subsub.occ>0)])<30:
            continue
        if len(subsub[(subsub.urban_areas<=0.05) & (subsub.cropland<=0.05) & (subsub.occ>0)])<30:
            continue
            
        if len(subsub)>1000: #.sort_values(by='obsvr_species_count',ascending=False)
            subsub = down_sample(subsub, samples_each_class=500)
            
        subsub['migration_type'] = flag
        
        #### if unique DOY<200: continue
        if subsub.DOY.unique().shape[0]<200:
            continue
        
        #### ml score
        x_names = ['DOY','urban_areas','cropland','duration_minutes']
        ml_score = get_ml_score(subsub[x_names], subsub['occ'])
        if ml_score>0.6:
            sp_df_list.append(subsub)
        else:
            continue
    
        #### print
        print(sp, year, 'Total: ',subsub.shape, 'Unique DOY: ', subsub.DOY.unique().shape[0],
              'Occ: ', subsub[subsub.occ>0].shape, 
              'Occ_urban: ',subsub[(subsub.urban_areas>0.05) & (subsub.occ>0)].shape,
              'Occ_cropland: ', subsub[(subsub.cropland>0.05) & (subsub.occ>0)].shape,
              'Occ_nature: ', subsub[(subsub.urban_areas<=0.05) & (subsub.cropland<=0.05) & (subsub.occ>0)].shape,
              'ML_score: ',ml_score)
        
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
    # df['duration_minutes'] = (df['duration_minutes'] - df['duration_minutes'].mean())/df['duration_minutes'].std()
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


# %%
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
passenger_df

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
breeding_df

# %%
wintering_df

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
    
    for year in sub.year.unique():
        try:
            subsub=sub[sub.year==year]

            X_train, X_test, y_train, y_test = train_test_split(subsub[[i for i in subsub.columns if not i=='occ']],
                                                                subsub['occ'],test_size=0.2)
            X_train_list.append(X_train)
            X_test_list.append(X_test)
            y_train_list.append(y_train)
            y_test_list.append(y_test)
        except Exception as e:
            print(e)
            continue

    
X_train = pd.concat(X_train_list, axis=0)
X_test = pd.concat(X_test_list, axis=0)
y_train = pd.concat(y_train_list, axis=0)
y_test = pd.concat(y_test_list, axis=0)

#### normalize
X_train_duration_minutes_std= X_train.duration_minutes.std()
X_train_duration_minutes_mean= X_train.duration_minutes.mean()
X_train_cropland_std= X_train.cropland.std()
X_train_cropland_mean= X_train.cropland.mean()
X_train_urban_areas_std= X_train.urban_areas.std()
X_train_urban_areas_mean= X_train.urban_areas.mean()

X_train['duration_minutes'] = (X_train['duration_minutes'] - X_train_duration_minutes_mean)/X_train_duration_minutes_std
X_test['duration_minutes'] = (X_test['duration_minutes'] - X_train_duration_minutes_mean)/X_train_duration_minutes_std
X_train['cropland'] = (X_train['cropland'] - X_train_cropland_mean)/X_train_cropland_std
X_test['cropland'] = (X_test['cropland'] - X_train_cropland_mean)/X_train_cropland_std
X_train['urban_areas'] = (X_train['urban_areas'] - X_train_urban_areas_mean)/X_train_urban_areas_std
X_test['urban_areas'] = (X_test['urban_areas'] - X_train_urban_areas_mean)/X_train_urban_areas_std



# %% [markdown]
# ## 5\.1 Define model

# %%
### passenger model
def cumulative_normal(x, mu, sigma, s=np.sqrt(2)):
    return 0.5 + 0.5 * pm.math.erf((x-mu)/(sigma*s))

### running

wintering_model = pm.Model()

with wintering_model:
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
    order_to_sp_obs = pm.Data('order_to_sp_obs', X_train[['sp_index','order_index']].drop_duplicates().sort_values(by='sp_index')['order_index'].values)
    
    ### prior
    ## spring departure
    spring_departure_dist_mu_overall = pm.TruncatedNormal('spring_departure_dist_mu_overall', mu=106, sigma=20, lower=0, upper=175)
    spring_departure_dist_mu_order_level = pm.TruncatedNormal('spring_departure_dist_mu_order_level', mu=spring_departure_dist_mu_overall, sigma=20, lower=0, upper=175, dims=('order_index'))
    spring_departure_dist_mu_sp_level = pm.TruncatedNormal('spring_departure_dist_mu_sp_level', mu=spring_departure_dist_mu_order_level[order_to_sp_obs], sigma=20, lower=0, upper=175, dims=('sp_index'))
    spring_departure_dist_mu = pm.TruncatedNormal('spring_departure_dist_mu', mu=spring_departure_dist_mu_sp_level, sigma=20, lower=0, upper=175, dims=('year_index','sp_index'))
    spring_departure_dist_sigma = pm.Gamma('spring_departure_dist_sigma', alpha=5, beta=0.5, dims=('year_index','sp_index'))
    
    ## fall arrival
    fall_arrival_dist_mu_overall = pm.TruncatedNormal('fall_arrival_dist_mu_overall', mu=291, sigma=20, lower=175, upper=366)
    fall_arrival_dist_mu_order_level = pm.TruncatedNormal('fall_arrival_dist_mu_order_level', mu=fall_arrival_dist_mu_overall, sigma=20, lower=175, upper=366, dims=('order_index'))
    fall_arrival_dist_mu_sp_level = pm.TruncatedNormal('fall_arrival_dist_mu_sp_level', mu=fall_arrival_dist_mu_order_level[order_to_sp_obs], sigma=20, lower=175, upper=366, dims=('sp_index'))
    fall_arrival_dist_mu = pm.TruncatedNormal('fall_arrival_dist_mu', mu=fall_arrival_dist_mu_sp_level, sigma=20, lower=175, upper=366, dims=('year_index','sp_index'))
    fall_arrival_dist_sigma = pm.Gamma('fall_arrival_dist_sigma', alpha=5, beta=0.5, dims=('year_index','sp_index'))
    
    ###
    simga_factor = 20
    effect_bound_fator = 60
    
    # #### urban effect   
    spring_departure_urban_effect_overall = pm.TruncatedNormal('spring_departure_urban_effect_overall', mu=0, sigma=simga_factor*X_train_urban_areas_std, upper=effect_bound_fator*X_train_urban_areas_std, lower=-effect_bound_fator*X_train_urban_areas_std)
    spring_departure_urban_effect_order_level = pm.TruncatedNormal('spring_departure_urban_effect_order_level', mu=spring_departure_urban_effect_overall, sigma=simga_factor*X_train_urban_areas_std, upper=effect_bound_fator*X_train_urban_areas_std, lower=-effect_bound_fator*X_train_urban_areas_std, dims=('order_index'))
    spring_departure_urban_effect_sp_level = pm.TruncatedNormal('spring_departure_urban_effect_sp_level', mu=spring_departure_urban_effect_order_level[order_to_sp_obs], sigma=simga_factor*X_train_urban_areas_std, upper=effect_bound_fator*X_train_urban_areas_std, lower=-effect_bound_fator*X_train_urban_areas_std, dims=('sp_index'))
    spring_departure_urban_effect = pm.TruncatedNormal('spring_departure_urban_effect', mu=spring_departure_urban_effect_sp_level,
                                              sigma=simga_factor*X_train_urban_areas_std, upper=effect_bound_fator*X_train_urban_areas_std, lower=-effect_bound_fator*X_train_urban_areas_std, dims=('year_index','sp_index'))
    
    fall_arrival_urban_effect_overall = pm.TruncatedNormal('fall_arrival_urban_effect_overall', mu=0, sigma=simga_factor*X_train_urban_areas_std, upper=effect_bound_fator*X_train_urban_areas_std, lower=-effect_bound_fator*X_train_urban_areas_std)
    fall_arrival_urban_effect_order_level = pm.TruncatedNormal('fall_arrival_urban_effect_order_level', mu=fall_arrival_urban_effect_overall, sigma=simga_factor*X_train_urban_areas_std, upper=effect_bound_fator*X_train_urban_areas_std, lower=-effect_bound_fator*X_train_urban_areas_std, dims=('order_index'))
    fall_arrival_urban_effect_sp_level = pm.TruncatedNormal('fall_arrival_urban_effect_sp_level', mu=fall_arrival_urban_effect_order_level[order_to_sp_obs], sigma=simga_factor*X_train_urban_areas_std, upper=effect_bound_fator*X_train_urban_areas_std, lower=-effect_bound_fator*X_train_urban_areas_std, dims=('sp_index'))
    fall_arrival_urban_effect = pm.TruncatedNormal('fall_arrival_urban_effect', mu=fall_arrival_urban_effect_sp_level,
                                          sigma=simga_factor*X_train_urban_areas_std, upper=effect_bound_fator*X_train_urban_areas_std, lower=-effect_bound_fator*X_train_urban_areas_std, dims=('year_index','sp_index'))

    
    #### cropland effect   
    spring_departure_cropland_effect_overall = pm.TruncatedNormal('spring_departure_cropland_effect_overall', mu=0, sigma=simga_factor*X_train_cropland_std, upper=effect_bound_fator*X_train_cropland_std, lower=-effect_bound_fator*X_train_cropland_std)
    spring_departure_cropland_effect_order_level = pm.TruncatedNormal('spring_departure_cropland_effect_order_level', mu=spring_departure_cropland_effect_overall, sigma=simga_factor*X_train_cropland_std, upper=effect_bound_fator*X_train_cropland_std, lower=-effect_bound_fator*X_train_cropland_std, dims=('order_index'))
    spring_departure_cropland_effect_sp_level = pm.TruncatedNormal('spring_departure_cropland_effect_sp_level', mu=spring_departure_cropland_effect_order_level[order_to_sp_obs], sigma=simga_factor*X_train_cropland_std, upper=effect_bound_fator*X_train_cropland_std, lower=-effect_bound_fator*X_train_cropland_std, dims=('sp_index'))
    spring_departure_cropland_effect = pm.TruncatedNormal('spring_departure_cropland_effect', mu=spring_departure_cropland_effect_sp_level, 
                                                sigma=simga_factor*X_train_cropland_std, upper=effect_bound_fator*X_train_cropland_std, lower=-effect_bound_fator*X_train_cropland_std, dims=('year_index', 'sp_index'))
    
    fall_arrival_cropland_effect_overall = pm.TruncatedNormal('fall_arrival_cropland_effect_overall', mu=0, sigma=simga_factor*X_train_cropland_std, upper=effect_bound_fator*X_train_cropland_std, lower=-effect_bound_fator*X_train_cropland_std)
    fall_arrival_cropland_effect_order_level = pm.TruncatedNormal('fall_arrival_cropland_effect_order_level', mu=fall_arrival_cropland_effect_overall, sigma=simga_factor*X_train_cropland_std, upper=effect_bound_fator*X_train_cropland_std, lower=-effect_bound_fator*X_train_cropland_std, dims=('order_index'))
    fall_arrival_cropland_effect_sp_level = pm.TruncatedNormal('fall_arrival_cropland_effect_sp_level', mu=fall_arrival_cropland_effect_order_level[order_to_sp_obs], sigma=simga_factor*X_train_cropland_std, upper=effect_bound_fator*X_train_cropland_std, lower=-effect_bound_fator*X_train_cropland_std, dims=('sp_index'))
    fall_arrival_cropland_effect = pm.TruncatedNormal('fall_arrival_cropland_effect', mu=fall_arrival_cropland_effect_sp_level, 
                                            sigma=simga_factor*X_train_cropland_std, upper=effect_bound_fator*X_train_cropland_std, lower=-effect_bound_fator*X_train_cropland_std, dims=('year_index', 'sp_index'))
    
    ##### intercept
    beta = pm.Normal('beta', mu=0, sigma=10, dims=('year_index', 'sp_index'))
    
    ### duration effect
    duration_effect_slope = pm.TruncatedNormal('duration_effect_slope', mu=0, sigma=10, lower=0, dims=('year_index','sp_index'))
    
    #### city effect modification on timing
    #####
    
    modified_spring_departure_dist = spring_departure_dist_mu[year_index_obs, sp_index_obs] + \
                                        (spring_departure_urban_effect[year_index_obs, sp_index_obs] * urban_areas_obs + \
                                            spring_departure_cropland_effect[year_index_obs, sp_index_obs] * cropland_obs)
    modified_fall_arrival_dist = fall_arrival_dist_mu[year_index_obs, sp_index_obs] + \
                                        (fall_arrival_urban_effect[year_index_obs, sp_index_obs] * urban_areas_obs + \
                                            fall_arrival_cropland_effect[year_index_obs, sp_index_obs] * cropland_obs)

    ### scaling factor
    w = pm.Dirichlet('w', a=np.ones([len(X_train.year_index.unique()),
                                    len(X_train.sp_index.unique()),
                                    2]))
    soft_diff_constraint_on_w = pm.Potential("soft_diff_constraint_on_w", -((pm.math.log(w[...,0]) - pm.math.log(w[...,1]))**2))
    
    ### variables times scalers
    spring_departure_effect_on_obs_prob = -(cumulative_normal(x=DOY_obs, 
                              mu=modified_spring_departure_dist, 
                              sigma=spring_departure_dist_sigma[year_index_obs, sp_index_obs])) * w[year_index_obs, sp_index_obs, 0]
 
        
    fall_arrival_effect_on_obs_prob =  (cumulative_normal(x=DOY_obs, 
                                                         mu=modified_fall_arrival_dist, 
                                                         sigma=fall_arrival_dist_sigma[year_index_obs, sp_index_obs]))* w[year_index_obs, sp_index_obs, 1]

    #### Effect of land use on detectability
    urban_effect_on_detectability_overall = pm.Normal('urban_effect_on_detectability_overall', mu=0, sigma=10, dims=('year_index','sp_index'))
    cropland_effect_on_detectability_overall = pm.Normal('cropland_effect_on_detectability_overall', mu=0, sigma=10, dims=('year_index','sp_index'))
    
    #### alpha
    alpha = pm.Lognormal('alpha', mu=2, sigma=1, dims=('year_index','sp_index'))
    
    #### linear term 
    y0 = beta[year_index_obs, sp_index_obs] + \
            duration_effect_slope[year_index_obs, sp_index_obs] * duration_obs + \
            urban_effect_on_detectability_overall[year_index_obs, sp_index_obs] * urban_areas_obs + \
            cropland_effect_on_detectability_overall[year_index_obs, sp_index_obs] * cropland_obs + \
            alpha[year_index_obs, sp_index_obs] * (spring_departure_effect_on_obs_prob + fall_arrival_effect_on_obs_prob)
            
    p_ = pm.math.sigmoid(y0)
    
    y_ = pm.Bernoulli("y_", p_, observed=occ_obs)
    


    

# %% [markdown]
# ## 5\.2 Sample chains

# %%
import pymc.sampling.jax as pmjax
import jax
import tensorflow_probability.substrates.jax as tfp
jax.scipy.special.erfcx = tfp.math.erfcx
from fastprogress.fastprogress import master_bar, progress_bar
from fastprogress.fastprogress import force_console_behavior
master_bar, progress_bar = force_console_behavior()

with wintering_model:
    idata = pmjax.sample_numpyro_nuts(SAMPLE_SIZE,chains=SAMPLE_CHAINS,cores=SAMPLE_CORES, tune=TUNES, progressbar=True)

    #  target_accept=0.99,
    #                   chains=2, cores=2, tune=3000, progressbar=True,
    #                     # trace = ['y_']
                      

# %%
with wintering_model:
    post_idata = pm.sample_posterior_predictive(idata)

# %%
pd.DataFrame({
    'DOY':X_train.DOY.values,
    'posetior_predictives':np.concatenate(post_idata.posterior_predictive['y_'], axis=0).mean(axis=0)
}).groupby('DOY').mean().plot()


# %% [markdown]
# ## 5\.3 Predict on test data

# %%
def pm_predict_winter_model(X_test):
    ### passenger model
    import math
    from scipy.special import expit
    def math_cumulative_normal(x, mu, sigma, s=np.sqrt(2)):
        return 0.5 + 0.5 * math.erf((x-mu)/(sigma*s))

    new_DOY_obs = X_test['DOY'].values
    new_sp_index_obs = X_test['sp_index'].values
    new_year_index_obs = X_test['year_index'].values
    new_urban_areas_obs = X_test['urban_areas'].values
    new_cropland_obs = X_test['cropland'].values
    new_duration_obs = X_test['duration_minutes'].values

    new_beta = np.concatenate(idata.posterior['beta'].values, axis=0).mean(axis=0)[new_year_index_obs, new_sp_index_obs]
    new_duration_effect_on_obs_prob = np.concatenate(idata.posterior['duration_effect_slope'].values, axis=0).mean(axis=0)[new_year_index_obs, new_sp_index_obs]

    new_modified_spring_departure_dist = np.concatenate(idata.posterior['spring_departure_dist_mu'], axis=0).mean(axis=0)[new_year_index_obs, new_sp_index_obs] + \
                                            np.concatenate(idata.posterior['spring_departure_urban_effect'], axis=0).mean(axis=0)[new_year_index_obs, new_sp_index_obs] * new_urban_areas_obs + \
                                                np.concatenate(idata.posterior['spring_departure_cropland_effect'], axis=0).mean(axis=0)[new_year_index_obs, new_sp_index_obs] * new_cropland_obs
                                                
    new_spring_departure_effect_on_obs_prob =  (
        np.array([-math_cumulative_normal(x=x,mu=mu,sigma=sigma) for x,mu,sigma in zip(
        new_DOY_obs,
        new_modified_spring_departure_dist, 
        np.concatenate(idata.posterior['spring_departure_dist_sigma'], axis=0).mean(axis=0)[new_year_index_obs, new_sp_index_obs]
    )])) * np.concatenate(idata.posterior['w'], axis=0).mean(axis=0)[new_year_index_obs, new_sp_index_obs, 0]

    new_modified_fall_arrival_dist = np.concatenate(idata.posterior['fall_arrival_dist_mu'], axis=0).mean(axis=0)[new_year_index_obs, new_sp_index_obs] + \
                                        np.concatenate(idata.posterior['fall_arrival_urban_effect'], axis=0).mean(axis=0)[new_year_index_obs, new_sp_index_obs] * new_urban_areas_obs + \
                                            np.concatenate(idata.posterior['fall_arrival_cropland_effect'], axis=0).mean(axis=0)[new_year_index_obs, new_sp_index_obs] * new_cropland_obs
                                            
    new_fall_arrival_effect_on_obs_prob = (np.array([
        math_cumulative_normal(x=x,mu=mu,sigma=sigma) for x,mu,sigma in zip(
            new_DOY_obs,
            new_modified_fall_arrival_dist,
            np.concatenate(idata.posterior['fall_arrival_dist_sigma'], axis=0).mean(axis=0)[new_year_index_obs, new_sp_index_obs]
        )
    ]))* np.concatenate(idata.posterior['w'], axis=0).mean(axis=0)[new_year_index_obs, new_sp_index_obs, 1]

    new_alpha = np.concatenate(idata.posterior['alpha'], axis=0).mean(axis=0)[new_year_index_obs, new_sp_index_obs]
    
    new_urban_effect_on_detectability_overall = np.concatenate(idata.posterior['urban_effect_on_detectability_overall'], axis=0).mean(axis=0)[new_year_index_obs, new_sp_index_obs]
    new_cropland_effect_on_detectability_overall = np.concatenate(idata.posterior['cropland_effect_on_detectability_overall'], axis=0).mean(axis=0)[new_year_index_obs, new_sp_index_obs]
    
    new_y0 = new_beta + \
            new_duration_effect_on_obs_prob * new_duration_obs + \
            new_urban_effect_on_detectability_overall * new_urban_areas_obs + \
            new_cropland_effect_on_detectability_overall * new_cropland_obs + \
            new_alpha * (new_spring_departure_effect_on_obs_prob + new_fall_arrival_effect_on_obs_prob)
            
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
# az.rcParams["plot.max_subplots"] = 50
# az.plot_trace(idata)
# plt.tight_layout()
# plt.show()


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
                 'test_f1':f1_score(y_test, y_pred),
                 
                 'X_train_duration_minutes_std':X_train_duration_minutes_std,
                 'X_train_duration_minutes_mean':X_train_duration_minutes_mean,
                 'X_train_cropland_std':X_train_cropland_std,
                 'X_train_cropland_mean':X_train_cropland_mean,
                 'X_train_urban_areas_std':X_train_urban_areas_std,
                 'X_train_urban_areas_mean':X_train_urban_areas_mean
                 }, f)


del idata, ess_, rhat_, bfmi_, post_idata, X_train, X_test, y_train, y_test #####################################


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
    
    for year in sub.year.unique():
        
        try:
            subsub=sub[sub.year==year]

            X_train, X_test, y_train, y_test = train_test_split(subsub[[i for i in subsub.columns if not i=='occ']],
                                                                subsub['occ'],test_size=0.2)
            X_train_list.append(X_train)
            X_test_list.append(X_test)
            y_train_list.append(y_train)
            y_test_list.append(y_test)
            
        except Exception as e:
            print(e)
            continue
    
    
X_train = pd.concat(X_train_list, axis=0)
X_test = pd.concat(X_test_list, axis=0)
y_train = pd.concat(y_train_list, axis=0)
y_test = pd.concat(y_test_list, axis=0)

#### normalize
X_train_duration_minutes_std= X_train.duration_minutes.std()
X_train_duration_minutes_mean= X_train.duration_minutes.mean()
X_train_cropland_std= X_train.cropland.std()
X_train_cropland_mean= X_train.cropland.mean()
X_train_urban_areas_std= X_train.urban_areas.std()
X_train_urban_areas_mean= X_train.urban_areas.mean()

X_train['duration_minutes'] = (X_train['duration_minutes'] - X_train_duration_minutes_mean)/X_train_duration_minutes_std
X_test['duration_minutes'] = (X_test['duration_minutes'] - X_train_duration_minutes_mean)/X_train_duration_minutes_std
X_train['cropland'] = (X_train['cropland'] - X_train_cropland_mean)/X_train_cropland_std
X_test['cropland'] = (X_test['cropland'] - X_train_cropland_mean)/X_train_cropland_std
X_train['urban_areas'] = (X_train['urban_areas'] - X_train_urban_areas_mean)/X_train_urban_areas_std
X_test['urban_areas'] = (X_test['urban_areas'] - X_train_urban_areas_mean)/X_train_urban_areas_std



# %% [markdown]
# ## 6\.1 Define model

# %%
### passenger model
def cumulative_normal(x, mu, sigma, s=np.sqrt(2)):
    return 0.5 + 0.5 * pm.math.erf((x-mu)/(sigma*s))

### running

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
    ## spring departure
    spring_arrival_dist_mu_overall = pm.TruncatedNormal('spring_arrival_dist_mu_overall', mu=106, sigma=20, lower=0, upper=175)
    spring_arrival_dist_mu_order_level = pm.TruncatedNormal('spring_arrival_dist_mu_order_level', mu=spring_arrival_dist_mu_overall, sigma=20, lower=0, upper=175, dims=('order_index'))
    spring_arrival_dist_mu_sp_level = pm.TruncatedNormal('spring_arrival_dist_mu_sp_level', mu=spring_arrival_dist_mu_order_level[order_to_sp_obs], sigma=20, lower=0, upper=175, dims=('sp_index'))
    spring_arrival_dist_mu = pm.TruncatedNormal('spring_arrival_dist_mu', mu=spring_arrival_dist_mu_sp_level, sigma=20, lower=0, upper=175, dims=('year_index','sp_index'))
    spring_arrival_dist_sigma = pm.Gamma('spring_arrival_dist_sigma', alpha=5, beta=0.5, dims=('year_index','sp_index'))
    
    ## fall arrival
    fall_departure_dist_mu_overall = pm.TruncatedNormal('fall_departure_dist_mu_overall', mu=291, sigma=20, lower=175, upper=366)
    fall_departure_dist_mu_order_level = pm.TruncatedNormal('fall_departure_dist_mu_order_level', mu=fall_departure_dist_mu_overall, sigma=20, lower=175, upper=366, dims=('order_index'))
    fall_departure_dist_mu_sp_level = pm.TruncatedNormal('fall_departure_dist_mu_sp_level', mu=fall_departure_dist_mu_order_level[order_to_sp_obs], sigma=20, lower=175, upper=366, dims=('sp_index'))
    fall_departure_dist_mu = pm.TruncatedNormal('fall_departure_dist_mu', mu=fall_departure_dist_mu_sp_level, sigma=20, lower=175, upper=366, dims=('year_index','sp_index'))
    fall_departure_dist_sigma = pm.Gamma('fall_departure_dist_sigma', alpha=5, beta=0.5, dims=('year_index','sp_index'))
    
    ###
    simga_factor = 20
    effect_bound_fator = 60
    
    # #### urban effect   
    spring_arrival_urban_effect_overall = pm.TruncatedNormal('spring_arrival_urban_effect_overall', mu=0, sigma=simga_factor*X_train_urban_areas_std, upper=effect_bound_fator*X_train_urban_areas_std, lower=-effect_bound_fator*X_train_urban_areas_std)
    spring_arrival_urban_effect_order_level = pm.TruncatedNormal('spring_arrival_urban_effect_order_level', mu=spring_arrival_urban_effect_overall, sigma=simga_factor*X_train_urban_areas_std, upper=effect_bound_fator*X_train_urban_areas_std, lower=-effect_bound_fator*X_train_urban_areas_std, dims=('order_index'))
    spring_arrival_urban_effect_sp_level = pm.TruncatedNormal('spring_arrival_urban_effect_sp_level', mu=spring_arrival_urban_effect_order_level[order_to_sp_obs], sigma=simga_factor*X_train_urban_areas_std, upper=effect_bound_fator*X_train_urban_areas_std, lower=-effect_bound_fator*X_train_urban_areas_std, dims=('sp_index'))
    spring_arrival_urban_effect = pm.TruncatedNormal('spring_arrival_urban_effect', mu=spring_arrival_urban_effect_sp_level,
                                              sigma=simga_factor*X_train_urban_areas_std, upper=effect_bound_fator*X_train_urban_areas_std, lower=-effect_bound_fator*X_train_urban_areas_std, dims=('year_index','sp_index'))
    
    fall_departure_urban_effect_overall = pm.TruncatedNormal('fall_departure_urban_effect_overall', mu=0, sigma=simga_factor*X_train_urban_areas_std, upper=effect_bound_fator*X_train_urban_areas_std, lower=-effect_bound_fator*X_train_urban_areas_std)
    fall_departure_urban_effect_order_level = pm.TruncatedNormal('fall_departure_urban_effect_order_level', mu=fall_departure_urban_effect_overall, sigma=simga_factor*X_train_urban_areas_std, upper=effect_bound_fator*X_train_urban_areas_std, lower=-effect_bound_fator*X_train_urban_areas_std, dims=('order_index'))
    fall_departure_urban_effect_sp_level = pm.TruncatedNormal('fall_departure_urban_effect_sp_level', mu=fall_departure_urban_effect_order_level[order_to_sp_obs], sigma=simga_factor*X_train_urban_areas_std, upper=effect_bound_fator*X_train_urban_areas_std, lower=-effect_bound_fator*X_train_urban_areas_std, dims=('sp_index'))
    fall_departure_urban_effect = pm.TruncatedNormal('fall_departure_urban_effect', mu=fall_departure_urban_effect_sp_level,
                                          sigma=simga_factor*X_train_urban_areas_std, upper=effect_bound_fator*X_train_urban_areas_std, lower=-effect_bound_fator*X_train_urban_areas_std, dims=('year_index','sp_index'))

    
    #### cropland effect   
    spring_arrival_cropland_effect_overall = pm.TruncatedNormal('spring_arrival_cropland_effect_overall', mu=0, sigma=simga_factor*X_train_cropland_std, upper=effect_bound_fator*X_train_cropland_std, lower=-effect_bound_fator*X_train_cropland_std)
    spring_arrival_cropland_effect_order_level = pm.TruncatedNormal('spring_arrival_cropland_effect_order_level', mu=spring_arrival_cropland_effect_overall, sigma=simga_factor*X_train_cropland_std, upper=effect_bound_fator*X_train_cropland_std, lower=-effect_bound_fator*X_train_cropland_std, dims=('order_index'))
    spring_arrival_cropland_effect_sp_level = pm.TruncatedNormal('spring_arrival_cropland_effect_sp_level', mu=spring_arrival_cropland_effect_order_level[order_to_sp_obs], sigma=simga_factor*X_train_cropland_std, upper=effect_bound_fator*X_train_cropland_std, lower=-effect_bound_fator*X_train_cropland_std, dims=('sp_index'))
    spring_arrival_cropland_effect = pm.TruncatedNormal('spring_arrival_cropland_effect', mu=spring_arrival_cropland_effect_sp_level, 
                                                sigma=simga_factor*X_train_cropland_std, upper=effect_bound_fator*X_train_cropland_std, lower=-effect_bound_fator*X_train_cropland_std, dims=('year_index', 'sp_index'))
    
    fall_departure_cropland_effect_overall = pm.TruncatedNormal('fall_departure_cropland_effect_overall', mu=0, sigma=simga_factor*X_train_cropland_std, upper=effect_bound_fator*X_train_cropland_std, lower=-effect_bound_fator*X_train_cropland_std)
    fall_departure_cropland_effect_order_level = pm.TruncatedNormal('fall_departure_cropland_effect_order_level', mu=fall_departure_cropland_effect_overall, sigma=simga_factor*X_train_cropland_std, upper=effect_bound_fator*X_train_cropland_std, lower=-effect_bound_fator*X_train_cropland_std, dims=('order_index'))
    fall_departure_cropland_effect_sp_level = pm.TruncatedNormal('fall_departure_cropland_effect_sp_level', mu=fall_departure_cropland_effect_order_level[order_to_sp_obs], sigma=simga_factor*X_train_cropland_std, upper=effect_bound_fator*X_train_cropland_std, lower=-effect_bound_fator*X_train_cropland_std, dims=('sp_index'))
    fall_departure_cropland_effect = pm.TruncatedNormal('fall_departure_cropland_effect', mu=fall_departure_cropland_effect_sp_level, 
                                            sigma=simga_factor*X_train_cropland_std, upper=effect_bound_fator*X_train_cropland_std, lower=-effect_bound_fator*X_train_cropland_std, dims=('year_index', 'sp_index'))
    
    ##### intercept
    beta = pm.Normal('beta', mu=0, sigma=10, dims=('year_index', 'sp_index'))
    
    ### duration effect
    duration_effect_slope = pm.TruncatedNormal('duration_effect_slope', mu=0, sigma=10, lower=0, dims=('year_index','sp_index'))
    
    #### city effect modification on timing
    #####
    
    modified_spring_arrival_dist = spring_arrival_dist_mu[year_index_obs, sp_index_obs] + \
                                        (spring_arrival_urban_effect[year_index_obs, sp_index_obs] * urban_areas_obs + \
                                            spring_arrival_cropland_effect[year_index_obs, sp_index_obs] * cropland_obs)
    modified_fall_departure_dist = fall_departure_dist_mu[year_index_obs, sp_index_obs] + \
                                        (fall_departure_urban_effect[year_index_obs, sp_index_obs] * urban_areas_obs + \
                                            fall_departure_cropland_effect[year_index_obs, sp_index_obs] * cropland_obs)

    ### scaling factor
    w = pm.Dirichlet('w', a=np.ones([len(X_train.year_index.unique()),
                                    len(X_train.sp_index.unique()),
                                    2]))
    soft_diff_constraint_on_w = pm.Potential("soft_diff_constraint_on_w", -((pm.math.log(w[...,0]) - pm.math.log(w[...,1]))**2))
    
    ### variables times scalers
    spring_arrival_effect_on_obs_prob = (cumulative_normal(x=DOY_obs, 
                              mu=modified_spring_arrival_dist, 
                              sigma=spring_arrival_dist_sigma[year_index_obs, sp_index_obs])) * w[year_index_obs, sp_index_obs, 0]
 
        
    fall_departure_effect_on_obs_prob =  -(cumulative_normal(x=DOY_obs, 
                                                         mu=modified_fall_departure_dist, 
                                                         sigma=fall_departure_dist_sigma[year_index_obs, sp_index_obs]))* w[year_index_obs, sp_index_obs, 1]

    #### Effect of land use on detectability
    urban_effect_on_detectability_overall = pm.Normal('urban_effect_on_detectability_overall', mu=0, sigma=10, dims=('year_index','sp_index'))
    cropland_effect_on_detectability_overall = pm.Normal('cropland_effect_on_detectability_overall', mu=0, sigma=10, dims=('year_index','sp_index'))
    
    #### alpha
    alpha = pm.Lognormal('alpha', mu=2, sigma=1, dims=('year_index','sp_index'))
    
    #### linear term 
    y0 = beta[year_index_obs, sp_index_obs] + \
            duration_effect_slope[year_index_obs, sp_index_obs] * duration_obs + \
            urban_effect_on_detectability_overall[year_index_obs, sp_index_obs] * urban_areas_obs + \
            cropland_effect_on_detectability_overall[year_index_obs, sp_index_obs] * cropland_obs + \
            alpha[year_index_obs, sp_index_obs] * (spring_arrival_effect_on_obs_prob + fall_departure_effect_on_obs_prob)
            
    p_ = pm.math.sigmoid(y0)
    
    y_ = pm.Bernoulli("y_", p_, observed=occ_obs)
    
    

# %% [markdown]
# ## 6\.2 Sample chains

# %%
import pymc.sampling.jax as pmjax
import jax
import tensorflow_probability.substrates.jax as tfp
jax.scipy.special.erfcx = tfp.math.erfcx
from fastprogress.fastprogress import master_bar, progress_bar
from fastprogress.fastprogress import force_console_behavior
master_bar, progress_bar = force_console_behavior()

with breeding_model:
    idata = pmjax.sample_numpyro_nuts(SAMPLE_SIZE,chains=SAMPLE_CHAINS,cores=SAMPLE_CORES, tune=TUNES, progressbar=True)
    
    #  target_accept=0.99,
    #                   chains=2, cores=2, tune=3000, progressbar=True,
    #                     # trace = ['y_']
                      

# %%
with breeding_model:
    post_idata = pm.sample_posterior_predictive(idata)

# %%
pd.DataFrame({
    'DOY':X_train.DOY.values,
    'posetior_predictives':np.concatenate(post_idata.posterior_predictive['y_'], axis=0).mean(axis=0)
}).groupby('DOY').mean().plot()


# %% [markdown]
# ## 6\.3 Predict on test data

# %%
def pm_predict_breeding_model(X_test):
    ### passenger model
    import math
    from scipy.special import expit
    def math_cumulative_normal(x, mu, sigma, s=np.sqrt(2)):
        return 0.5 + 0.5 * math.erf((x-mu)/(sigma*s))

    new_DOY_obs = X_test['DOY'].values
    new_sp_index_obs = X_test['sp_index'].values
    new_year_index_obs = X_test['year_index'].values
    new_urban_areas_obs = X_test['urban_areas'].values
    new_cropland_obs = X_test['cropland'].values
    new_duration_obs = X_test['duration_minutes'].values

    new_beta = np.concatenate(idata.posterior['beta'].values, axis=0).mean(axis=0)[new_year_index_obs, new_sp_index_obs]
    new_duration_effect_on_obs_prob = np.concatenate(idata.posterior['duration_effect_slope'].values, axis=0).mean(axis=0)[new_year_index_obs, new_sp_index_obs]

    new_modified_spring_arrival_dist = np.concatenate(idata.posterior['spring_arrival_dist_mu'], axis=0).mean(axis=0)[new_year_index_obs, new_sp_index_obs] + \
                                            np.concatenate(idata.posterior['spring_arrival_urban_effect'], axis=0).mean(axis=0)[new_year_index_obs, new_sp_index_obs] * new_urban_areas_obs + \
                                                np.concatenate(idata.posterior['spring_arrival_cropland_effect'], axis=0).mean(axis=0)[new_year_index_obs, new_sp_index_obs] * new_cropland_obs
                                                
    new_spring_arrival_effect_on_obs_prob =  (
        np.array([math_cumulative_normal(x=x,mu=mu,sigma=sigma) for x,mu,sigma in zip(
        new_DOY_obs,
        new_modified_spring_arrival_dist, 
        np.concatenate(idata.posterior['spring_arrival_dist_sigma'], axis=0).mean(axis=0)[new_year_index_obs, new_sp_index_obs]
    )])) * np.concatenate(idata.posterior['w'], axis=0).mean(axis=0)[new_year_index_obs, new_sp_index_obs, 0]

    new_modified_fall_departure_dist = np.concatenate(idata.posterior['fall_departure_dist_mu'], axis=0).mean(axis=0)[new_year_index_obs, new_sp_index_obs] + \
                                        np.concatenate(idata.posterior['fall_departure_urban_effect'], axis=0).mean(axis=0)[new_year_index_obs, new_sp_index_obs] * new_urban_areas_obs + \
                                            np.concatenate(idata.posterior['fall_departure_cropland_effect'], axis=0).mean(axis=0)[new_year_index_obs, new_sp_index_obs] * new_cropland_obs
                                            
    new_fall_departure_effect_on_obs_prob = (np.array([
        -math_cumulative_normal(x=x,mu=mu,sigma=sigma) for x,mu,sigma in zip(
            new_DOY_obs,
            new_modified_fall_departure_dist,
            np.concatenate(idata.posterior['fall_departure_dist_sigma'], axis=0).mean(axis=0)[new_year_index_obs, new_sp_index_obs]
        )
    ]))* np.concatenate(idata.posterior['w'], axis=0).mean(axis=0)[new_year_index_obs, new_sp_index_obs, 1]

    new_alpha = np.concatenate(idata.posterior['alpha'], axis=0).mean(axis=0)[new_year_index_obs, new_sp_index_obs]
    
    new_urban_effect_on_detectability_overall = np.concatenate(idata.posterior['urban_effect_on_detectability_overall'], axis=0).mean(axis=0)[new_year_index_obs, new_sp_index_obs]
    new_cropland_effect_on_detectability_overall = np.concatenate(idata.posterior['cropland_effect_on_detectability_overall'], axis=0).mean(axis=0)[new_year_index_obs, new_sp_index_obs]
    
    new_y0 = new_beta + \
            new_duration_effect_on_obs_prob * new_duration_obs + \
            new_urban_effect_on_detectability_overall * new_urban_areas_obs + \
            new_cropland_effect_on_detectability_overall * new_cropland_obs + \
            new_alpha * (new_spring_arrival_effect_on_obs_prob + new_fall_departure_effect_on_obs_prob)
            
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
az.rcParams["plot.max_subplots"] = 50
az.plot_trace(idata)
plt.tight_layout()
plt.show()

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
                 'test_f1':f1_score(y_test, y_pred),
                 
                 'X_train_duration_minutes_std':X_train_duration_minutes_std,
                 'X_train_duration_minutes_mean':X_train_duration_minutes_mean,
                 'X_train_cropland_std':X_train_cropland_std,
                 'X_train_cropland_mean':X_train_cropland_mean,
                 'X_train_urban_areas_std':X_train_urban_areas_std,
                 'X_train_urban_areas_mean':X_train_urban_areas_mean
                 }, f)



# %%
rhat_

# %%
bfmi_

# %%


# %%


# %%


# %%


# %%


# %%



