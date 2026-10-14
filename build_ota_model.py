"""DemandSense synthetic OTA model pipeline.
All OTA observations are synthetic demonstration data, not real platform performance.
Run: python build_ota_model.py
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, roc_auc_score, brier_score_loss

BASE = Path(__file__).resolve().parent
RNG_SEED = 42
CHANNELS = {
    "Direct Website": {"commission_pct": 0, "acquisition_cost": 200, "base_conversion": 0.15, "cancel_base": .07, "discount_pct": 0},
    "MakeMyTrip": {"commission_pct": 18, "acquisition_cost": 0, "base_conversion": .19, "cancel_base": .17, "discount_pct": 3},
    "Goibibo": {"commission_pct": 18, "acquisition_cost": 0, "base_conversion": .16, "cancel_base": .16, "discount_pct": 3},
    "Booking.com": {"commission_pct": 17, "acquisition_cost": 0, "base_conversion": .14, "cancel_base": .19, "discount_pct": 2},
    "Agoda": {"commission_pct": 18, "acquisition_cost": 0, "base_conversion": .12, "cancel_base": .15, "discount_pct": 2},
    "Expedia": {"commission_pct": 18, "acquisition_cost": 0, "base_conversion": .10, "cancel_base": .18, "discount_pct": 2},
}
CAT = ["channel", "season"]
NUM = ["day_of_week", "month", "is_weekend", "is_event", "demand_index", "room_rate", "discount_pct", "commission_pct", "lead_days"]
FEATURES = CAT + NUM

def make_data():
    rng = np.random.default_rng(RNG_SEED)
    dates = pd.date_range("2023-01-01", "2026-10-12", freq="D")
    rows=[]
    for date in dates:
        dow=date.dayofweek; month=date.month
        season = "peak" if month in [4,5,6,10,11,12] else ("shoulder" if month in [2,3,7,9] else "low")
        weekend = int(dow >= 5)
        event = int((month,dow) in [(10,5),(12,5),(4,5)] and rng.random()<.18 or rng.random()<.025)
        # synthetic demand index (0-100); this is a generator assumption, not observed tourism data
        demand = 48 + 13*weekend + (12 if season=="peak" else -7 if season=="low" else 0) + 15*event + 8*np.sin(2*np.pi*date.dayofyear/365.25) + rng.normal(0,8)
        demand=float(np.clip(demand,10,98))
        rate=float(np.clip(3000 + 14*demand + (350 if weekend else 0) + rng.normal(0,180),1800,6500))
        for ch,terms in CHANNELS.items():
            discount=terms['discount_pct']
            # Synthetic response formula; intentionally documented so users know what model can learn
            log_demand = (demand-50)/22 + (terms['base_conversion']-.14)*5 + weekend*.08 + event*.20 - discount*.025
            expected = max(.15, 3.0*np.exp(log_demand))
            bookings=int(rng.poisson(expected))
            lead=int(rng.integers(1,46))
            cancel_p=float(np.clip(terms['cancel_base'] + max(0,lead-14)*.0012 + discount*.003 - event*.025, .02,.55))
            cancelled=int(rng.binomial(bookings,cancel_p))
            rows.append({"date":date,"channel":ch,"season":season,"day_of_week":dow,"month":month,
                         "is_weekend":weekend,"is_event":event,"demand_index":round(demand,2),
                         "room_rate":round(rate,2),"discount_pct":discount,"commission_pct":terms['commission_pct'],
                         "lead_days":lead,"bookings":bookings,"cancelled_bookings":cancelled,
                         "synthetic_data":True})
    df=pd.DataFrame(rows)
    df.to_csv(BASE/"demandsense_synthetic_ota_data.csv",index=False)
    return df

def make_pipeline(model, cat=CAT, num=NUM):
    prep=ColumnTransformer([("cat",OneHotEncoder(handle_unknown="ignore"),cat),("num","passthrough",num)])
    return Pipeline([("prep",prep),("model",model)])

def main():
    df=make_data()
    # Strict chronological holdout: the final 20% of unique dates are never used in fitting.
    unique_dates=np.sort(df.date.unique()); cutoff=unique_dates[int(len(unique_dates)*.8)]
    train=df[df.date<cutoff].copy(); test=df[df.date>=cutoff].copy()
    Xtr=train[FEATURES]; Xte=test[FEATURES]
    book_model=make_pipeline(RandomForestRegressor(n_estimators=250,min_samples_leaf=3,random_state=RNG_SEED,n_jobs=-1))
    book_model.fit(Xtr,train.bookings)
    pred=np.maximum(0,book_model.predict(Xte))
    # simple baseline: mean booking count for each channel in training set
    channel_mean=train.groupby('channel').bookings.mean()
    base=np.array([channel_mean.get(c,train.bookings.mean()) for c in test.channel])
    mae=float(mean_absolute_error(test.bookings,pred)); rmse=float(np.sqrt(mean_squared_error(test.bookings,pred)))
    base_mae=float(mean_absolute_error(test.bookings,base)); base_rmse=float(np.sqrt(mean_squared_error(test.bookings,base)))
    # Cancellation probability trained only on booking rows with at least one booking; target is cancelled / bookings.
    ctrain=train[train.bookings>0].copy(); ctest=test[test.bookings>0].copy()
    ctrain['cancelled_any']=(ctrain.cancelled_bookings/ctrain.bookings >= .5).astype(int)
    ctest['cancelled_any']=(ctest.cancelled_bookings/ctest.bookings >= .5).astype(int)
    cancel_model=make_pipeline(RandomForestClassifier(n_estimators=250,min_samples_leaf=4,class_weight='balanced',random_state=RNG_SEED,n_jobs=-1))
    cancel_model.fit(ctrain[FEATURES],ctrain.cancelled_any)
    cp=cancel_model.predict_proba(ctest[FEATURES])[:,1]
    y=ctest.cancelled_any.to_numpy()
    cancel_metrics={"accuracy":float(accuracy_score(y,(cp>=.5).astype(int))),"brier_score":float(brier_score_loss(y,cp))}
    if len(np.unique(y))==2: cancel_metrics['roc_auc']=float(roc_auc_score(y,cp))
    joblib.dump(book_model,BASE/"demandsense_ota_bookings_model.joblib")
    joblib.dump(cancel_model,BASE/"demandsense_ota_cancellation_model.joblib")
    metrics={"data_type":"SYNTHETIC DEMONSTRATION ONLY","rows":int(len(df)),"dates":int(df.date.nunique()),"channels":list(CHANNELS),
      "train_start":str(train.date.min().date()),"train_end":str(train.date.max().date()),"test_start":str(test.date.min().date()),"test_end":str(test.date.max().date()),
      "booking_model":{"mae_bookings_per_channel_day":mae,"rmse_bookings_per_channel_day":rmse,"baseline_channel_mean_mae":base_mae,"baseline_channel_mean_rmse":base_rmse,"beats_baseline_mae":bool(mae<base_mae)},
      "cancellation_classifier":cancel_metrics,
      "limitations":["All OTA records and response patterns are synthetic assumptions.","Metrics validate performance only against the synthetic generator, not real OTA behavior.","Replace with anonymised hotel/channel booking records before making real business claims."]}
    (BASE/"demandsense_ota_model_metrics.json").write_text(json.dumps(metrics,indent=2))
    # Generate example predictions for last 30 dates in held-out set, preserving test rows/features.
    demo=test.copy(); demo['predicted_bookings']=np.maximum(0,book_model.predict(demo[FEATURES]))
    demo['predicted_cancellation_probability']=cancel_model.predict_proba(demo[FEATURES])[:,1]
    demo.to_csv(BASE/"demandsense_ota_holdout_predictions.csv",index=False)
    print(json.dumps(metrics,indent=2))

if __name__=='__main__': main()
