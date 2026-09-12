
import io
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

st.set_page_config(page_title="AquaClass Global 2.0", page_icon="💧", layout="wide")

PARAMETERS = [
    "pH","Turbidity","TDS","Dissolved Oxygen","Nitrate","BOD","COD",
    "Conductivity","Coliform Indicator","Temperature"
]
ALIASES = {
    "pH":["ph","p_h"],
    "Turbidity":["turbidity","ntu","turbidity_ntu"],
    "TDS":["tds","total_dissolved_solids","total dissolved solids"],
    "Dissolved Oxygen":["dissolved_oxygen","dissolved oxygen","do","oxygen"],
    "Nitrate":["nitrate","nitrate_mg_l","no3","no3_n"],
    "BOD":["bod","biochemical_oxygen_demand"],
    "COD":["cod","chemical_oxygen_demand"],
    "Conductivity":["conductivity","electrical_conductivity","ec"],
    "Coliform Indicator":["coliform","coliform_indicator","total_coliform","fecal_coliform"],
    "Temperature":["temperature","temp","water_temperature"]
}
DEFAULT_BANDS = {
    "pH":(6.5,8.5,6.0,9.0,5.5,9.5),
    "Turbidity":(1,5,10),
    "TDS":(300,600,1000),
    "Dissolved Oxygen":(6,4,2),
    "Nitrate":(10,25,50),
    "BOD":(3,6,10),
    "COD":(20,50,100),
    "Conductivity":(500,1000,2000),
    "Coliform Indicator":(1,10,100),
    "Temperature":(25,30,35)
}
LOWER_BETTER = {"Turbidity","TDS","Nitrate","BOD","COD","Conductivity","Coliform Indicator"}
HIGHER_BETTER = {"Dissolved Oxygen"}

def norm(x):
    return "".join(c for c in str(x).lower().strip() if c.isalnum())

def standardize(df):
    df=df.copy()
    normalized={norm(c):c for c in df.columns}
    rename={}
    for p in PARAMETERS:
        for candidate in [p]+ALIASES[p]:
            if norm(candidate) in normalized:
                rename[normalized[norm(candidate)]]=p
                break
    return df.rename(columns=rename)

def numeric(df):
    for p in PARAMETERS:
        if p in df:
            df[p]=pd.to_numeric(df[p],errors="coerce")
    for c in ["Latitude","Longitude","Year"]:
        if c in df:
            df[c]=pd.to_numeric(df[c],errors="coerce")
    return df

def cls(p,v,b):
    if pd.isna(v): return "Not available"
    v=float(v)
    if p=="pH":
        if b[0]<=v<=b[1]: return "Good"
        if b[2]<=v<=b[3]: return "Moderate"
        if b[4]<=v<=b[5]: return "Poor"
        return "Very Poor"
    if p=="Dissolved Oxygen":
        if v>=b[0]: return "Good"
        if v>=b[1]: return "Moderate"
        if v>=b[2]: return "Poor"
        return "Very Poor"
    if p=="Temperature":
        if 5<=v<=b[0]: return "Good"
        if v<=b[1]: return "Moderate"
        if v<=b[2]: return "Poor"
        return "Very Poor"
    if v<=b[0]: return "Good"
    if v<=b[1]: return "Moderate"
    if v<=b[2]: return "Poor"
    return "Very Poor"

def score(label):
    return {"Good":4,"Moderate":3,"Poor":2,"Very Poor":1}.get(label,np.nan)

def overall(row,bands):
    s=[score(cls(p,row[p],bands[p])) for p in PARAMETERS if p in row and not pd.isna(row[p])]
    s=[x for x in s if not pd.isna(x)]
    if not s:return "Not available"
    a=np.mean(s)
    return "Good" if a>=3.5 else "Moderate" if a>=2.5 else "Poor" if a>=1.5 else "Very Poor"

def detect_target(df):
    for c in ["Water-quality classifier","Water Quality Class","water_quality_class","class","label","target"]:
        if c in df:return c
    return None

def train(df):
    target=detect_target(df)
    if not target:return None,None,"No label column found."
    x=df[PARAMETERS].copy()
    y=df[target].astype(str)
    valid=x.notna().sum(axis=1)>=3
    x=x.loc[valid].copy(); y=y.loc[valid]
    if len(x)<30 or y.nunique()<2:return None,None,"Need at least 30 usable labeled rows and 2 classes."
    x=x.fillna(x.median(numeric_only=True))
    try:
        xt,xv,yt,yv=train_test_split(x,y,test_size=.2,random_state=42,stratify=y)
    except ValueError:
        xt,xv,yt,yv=train_test_split(x,y,test_size=.2,random_state=42)
    m=RandomForestClassifier(n_estimators=400,class_weight="balanced",random_state=42,n_jobs=-1)
    m.fit(xt,yt); pred=m.predict(xv)
    return m,classification_report(yv,pred,output_dict=True,zero_division=0),None

st.title("💧 AquaClass Global 2.0")
st.caption("Explainable water-quality screening, historical analysis and worldwide spatial exploration")

with st.sidebar:
    st.header("Data")
    files=st.file_uploader("Upload one or more CSV files",type=["csv"],accept_multiple_files=True)
    source_mode=st.radio("Analysis mode",["Uploaded data","Demo data"])
    st.divider()
    st.header("Filters")
    country_filter=None
    year_range=None

if source_mode=="Demo data" and not files:
    data=pd.DataFrame([
        ["Pakistan","Faisalabad",31.45,73.14,2024,7.2,2.1,420,5.2,14,4.5,35,720,4,27],
        ["India","Delhi",28.61,77.21,2023,7.8,8.2,680,3.1,32,7.1,88,1200,18,29],
        ["Germany","Berlin",52.52,13.40,2022,7.4,1.0,290,8.2,5,1.5,15,450,1,18],
        ["Brazil","Manaus",-3.12,-60.02,2021,6.9,4.5,350,6.1,11,3.8,29,620,5,26],
        ["United States","Mississippi",32.35,-90.88,2020,7.0,3.2,510,5.7,18,4.9,42,800,7,24],
    ],columns=["Country","Location","Latitude","Longitude","Year"]+PARAMETERS)
    source="Demo records"
elif files:
    frames=[]
    for f in files:
        try:
            x=pd.read_csv(f); x["__source_file"]=f.name; frames.append(x)
        except Exception as e: st.error(f"{f.name}: {e}")
    data=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()
    source=f"{len(files)} uploaded file(s)"
else:
    data=pd.DataFrame(); source="No data"

if data.empty:
    st.info("Upload one or more CSV files to begin.")
    st.stop()

data=standardize(data); data=numeric(data)
missing=[p for p in PARAMETERS if p not in data.columns]
if missing:
    st.warning("Missing columns: "+", ".join(missing))

# Configurable screening bands
bands={}
with st.sidebar.expander("Screening thresholds",expanded=False):
    st.caption("Generic screening defaults. Replace with a documented standard for your intended use.")
    for p in PARAMETERS:
        b=DEFAULT_BANDS[p]
        if p=="pH":
            bands[p]=(st.number_input(f"{p} good min",value=float(b[0]),key="ph1"),
                      st.number_input(f"{p} good max",value=float(b[1]),key="ph2"),
                      st.number_input(f"{p} moderate min",value=float(b[2]),key="ph3"),
                      st.number_input(f"{p} moderate max",value=float(b[3]),key="ph4"),
                      st.number_input(f"{p} poor min",value=float(b[4]),key="ph5"),
                      st.number_input(f"{p} poor max",value=float(b[5]),key="ph6"))
        elif p=="Dissolved Oxygen":
            bands[p]=(st.number_input(f"{p} good ≥",value=float(b[0]),key=p+"1"),
                      st.number_input(f"{p} moderate ≥",value=float(b[1]),key=p+"2"),
                      st.number_input(f"{p} poor ≥",value=float(b[2]),key=p+"3"))
        else:
            bands[p]=(st.number_input(f"{p} good upper",value=float(b[0]),key=p+"1"),
                      st.number_input(f"{p} moderate upper",value=float(b[1]),key=p+"2"),
                      st.number_input(f"{p} poor upper",value=float(b[2]),key=p+"3"))

for p in PARAMETERS:
    if p not in data: continue
    data[p]=pd.to_numeric(data[p],errors="coerce")
data["Water-quality classifier"]=data.apply(lambda r: overall(r,bands),axis=1)

# filters
if "Country" in data:
    countries=["All"]+sorted(data["Country"].dropna().astype(str).unique().tolist())
    chosen=st.sidebar.selectbox("Country",countries)
    if chosen!="All": data=data[data["Country"].astype(str)==chosen]
if "Year" in data and data["Year"].notna().any():
    mn=int(data["Year"].min()); mx=int(data["Year"].max())
    if mn<mx:
        yr=st.sidebar.slider("Year range",mn,mx,(mn,mx))
        data=data[data["Year"].between(*yr)]

tab1,tab2,tab3,tab4,tab5,tab6=st.tabs([
    "Dashboard","Record explanation","World map","Time series","Historical issues","ML"
])

with tab1:
    st.subheader("Classification results")
    counts=data["Water-quality classifier"].value_counts()
    cs=st.columns(4)
    for c,l in zip(cs,["Good","Moderate","Poor","Very Poor"]):
        c.metric(l,int(counts.get(l,0)))
    cols=[c for c in ["Country","Location","Latitude","Longitude","Year"]+PARAMETERS+["Water-quality classifier"] if c in data]
    st.dataframe(data[cols],use_container_width=True,hide_index=True)
    st.download_button("Download analyzed CSV",data.to_csv(index=False).encode(),"aquaclass_analyzed.csv","text/csv")

with tab2:
    st.subheader("Explain one observation")
    i=st.number_input("Record",0,max(0,len(data)-1),0)
    row=data.iloc[int(i)]
    st.markdown(f"## Final result: **{row['Water-quality classifier']}**")
    exp=[]
    for p in PARAMETERS:
        if p in row:
            label=cls(p,row[p],bands[p])
            exp.append([p,row[p],label,score(label)])
    e=pd.DataFrame(exp,columns=["Parameter","Value","Class","Score"])
    st.dataframe(e,use_container_width=True,hide_index=True)
    fig=px.bar(e.dropna(subset=["Score"]).sort_values("Score"),x="Score",y="Parameter",color="Class",orientation="h",
               title="Parameter screening contribution (transparent 1–4 score)")
    st.plotly_chart(fig,use_container_width=True)
    st.caption("The screening score indicates how each parameter falls within the configured bands. It is not a universal regulatory index.")

with tab3:
    st.subheader("Worldwide water-quality observations")
    if {"Latitude","Longitude"}.issubset(data.columns):
        m=data.dropna(subset=["Latitude","Longitude"]).copy()
        if len(m):
            fig=px.scatter_geo(m,lat="Latitude",lon="Longitude",color="Water-quality classifier",
                               hover_name="Location" if "Location" in m else None,
                               hover_data=[c for c in ["Country","Year"]+PARAMETERS if c in m],
                               projection="natural earth",title="Uploaded observations")
            st.plotly_chart(fig,use_container_width=True)
        else: st.info("No valid coordinates.")
    else: st.info("Latitude and Longitude are required for mapping.")

with tab4:
    st.subheader("Historical / temporal analysis")
    if "Year" in data.columns and data["Year"].notna().any():
        ts=data.groupby(["Year","Water-quality classifier"]).size().reset_index(name="Records")
        fig=px.line(ts,x="Year",y="Records",color="Water-quality classifier",markers=True)
        st.plotly_chart(fig,use_container_width=True)
        if "Country" in data:
            country_year=data.groupby(["Year","Country"]).size().reset_index(name="Records")
            fig2=px.density_heatmap(country_year,x="Year",y="Country",z="Records",title="Observation coverage by country and year")
            st.plotly_chart(fig2,use_container_width=True)
    else: st.info("Add a Year column to analyze trends.")

with tab5:
    st.subheader("Global historical water-issue reference")
    hist=Path(__file__).parent/"historical_water_issues.csv"
    if hist.exists():
        h=pd.read_csv(hist)
        st.dataframe(h,use_container_width=True,hide_index=True)
        fig=px.scatter_geo(h,lat="Latitude",lon="Longitude",color="Issue Type",
                           hover_name="Event",hover_data=["Country","Year","Severity"],
                           projection="natural earth",title="Reference locations of documented water-related issues")
        st.plotly_chart(fig,use_container_width=True)
        st.caption("This reference layer is illustrative. Use authoritative event databases and cite the source for research/publication use.")
    else: st.warning("Historical reference file is missing.")

with tab6:
    st.subheader("Optional supervised ML model")
    if not files:
        st.info("Upload a labeled CSV containing a target such as `Water-quality classifier`.")
    else:
        model,report,err=train(data)
        if err: st.warning(err)
        else:
            st.success("Random Forest trained.")
            st.metric("Validation accuracy",f"{report['accuracy']:.1%}")
            st.dataframe(pd.DataFrame(report).T,use_container_width=True)
            imp=pd.Series(model.feature_importances_,index=PARAMETERS).sort_values()
            fig=px.bar(imp,x=imp.values,y=imp.index,orientation="h",labels={"x":"Importance","y":"Parameter"},
                       title="Global model feature importance")
            st.plotly_chart(fig,use_container_width=True)

st.divider()
st.caption("AquaClass is decision-support software. It does not replace accredited laboratory testing, regulatory standards, or public-health assessment.")
