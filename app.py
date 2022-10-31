import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
#from sklearn.ensemble import RandomForestClassifier
import pickle
st.set_page_config(layout="wide")

@st.cache(show_spinner=False)
def load_setting():
    settings = {
        'Age': {'values': [0, 100], 'type': 'slider', 'init_value': 55, 'add_after': ', year'},
        'Cardiopulmonary bypass time': {'values': [0, 600], 'type': 'slider', 'init_value': 40, 'add_after': ', min'},
        'Aortic cross clamp time': {'values': [0, 600], 'type': 'slider', 'init_value': 20, 'add_after': ', min'},
        'Diabetes': {'values': ["No", "Yes"], 'type': 'selectbox', 'init_value': 0, 'add_after': ''},
        'Preoperative creatinine value': {'values': [0, 2000], 'type': 'slider', 'init_value': 80,'add_after': ', μmoI/L'},
        'Preoperative renal insufficiency': {'values': ["No", "Yes"], 'type': 'selectbox', 'init_value': 0,'add_after': ''},
        'Preoperative arrhythmia': {'values': ["No", "Yes"], 'type': 'selectbox', 'init_value': 0, 'add_after': ''},
        'COPD': {'values': ["No", "Yes"], 'type': 'selectbox', 'init_value': 0, 'add_after': ''},
        'Renal artery stenosis': {'values': ["No", "Yes"], 'type': 'selectbox', 'init_value': 0, 'add_after': ''},
        'Preoperative Mechanical ventilation time': {'values': [0, 1000], 'type': 'slider', 'init_value': 40,'add_after': ', hour'},
        'IABP': {'values': ["No", "Yes"], 'type': 'selectbox', 'init_value': 0, 'add_after': ''}
    }
    input_keys = list(settings.keys())
    return settings, input_keys


settings, input_keys = load_setting()


@st.cache(allow_output_mutation=True)
def get_model():
    model = pickle.load(open('./Models/rand_classifier.pickle', 'rb'))
    scaler = pickle.load(open('./Models/scaler.pkl', 'rb'))
    return model,scaler


def get_code():
    sidebar_code = []
    for key in settings:
        if settings[key]['type'] == 'slider':
            sidebar_code.append(
                "{} = st.slider('{}',{},{},key='{}')".format(
                    key.replace(' ', '____'),
                    key + settings[key]['add_after'],
                    # settings[key]['values'][0],
                    ','.join(['{}'.format(value) for value in settings[key]['values']]),
                    settings[key]['init_value'],
                    key
                )
            )
        if settings[key]['type'] == 'selectbox':
            sidebar_code.append('{} = st.selectbox("{}",({}),{},key="{}")'.format(
                key.replace(' ', '____'),
                key + settings[key]['add_after'],
                ','.join('"{}"'.format(value) for value in settings[key]['values']),
                settings[key]['init_value'],
                key
            )
            )
    return sidebar_code




# print('\n'.join(sidebar_code))
if 'patients' not in st.session_state:
    st.session_state['patients'] = []
if 'display' not in st.session_state:
    st.session_state['display'] = 1
if 'model' not in st.session_state:
    st.session_state['model'] = 'rand_classifier.pickle'
best_model,scaler = get_model()
sidebar_code = get_code()
def plot_prediction():
    print(st.session_state['patients'])
    pd_data = pd.concat(
        [
            pd.DataFrame(
                {
                    'Probability': [item['prob']],
                    'outcome': [item['outcome']],
                    'Patient': [item['No']]
                }
            ) for item in st.session_state['patients']
        ]
    )
    if st.session_state['display']:
        fig =px.scatter(pd_data, x="Patient", y="Probability",size='Probability',color="Probability",range_y=[0, 1.0],range_x=[0, pd_data.shape[0]+10])
    else:
        fig = px.scatter(pd_data.loc[pd_data['Patient'] == pd_data['Patient'].to_list()[-1], :],
                         x="Patient", y="Probability",size='Probability',color="Probability",range_y=[0, 1.0])
    fig.update_layout(xaxis={'dtick': 1},yaxis={'dtick': 0.2})
    fig.update_coloraxes(cmax= 1,cmin=0)
    fig.update_xaxes(visible=False)
    fig.update_layout(template='simple_white',
                      title={
                          'text': 'Estimated probability of postoperative complications',
                          'y': 1.0,
                          'x': 0.5,
                          'xanchor': 'center',
                          'yanchor': 'top',
                          'font': dict(
                              size=25
                          )
                      },
                      plot_bgcolor="white",
                      xaxis_title="Patients",
                      yaxis_title="Estimated probability",
                      coloraxis=dict(
                          colorscale = [[0, 'rgb(51,255,51)'], [1, 'rgb(247,77,80)']]
                      )

                      )
    st.plotly_chart(fig, use_container_width=True)


def plot_patients():
    patients = pd.concat(
        [
            pd.DataFrame(
                dict(
                    {
                        'Patients': [item['No']],
                        'Probability': ["{:.2f}%".format(item['prob'] * 100)],
                        'Prediction': [item['outcome']]
                    },
                    **item['arg']
                )
            ) for item in st.session_state['patients']
        ]
    ).reset_index(drop=True)
    st.dataframe(patients)

# @st.cache(show_spinner=True)
def predict():
    print('update patients . ##########')
    #print(st.session_state)
    input = []
    for key in input_keys:
        value = st.session_state[key]
        if isinstance(value, int):
            input.append(value)
        if isinstance(value, str):
            input.append(settings[key]['values'].index(value))
    df = pd.DataFrame(columns=['Age','Tiwaishijian','zhudongmaizuduan',
                               'tangniaobing','shuqianjigan','shuqianshengongnengbuquan',
                               'shuqianxinlvshichang','COPD','shendongmaixiazhai',
                               'huxijifuzhushijian','IABP'])
    df.loc[0] = input
    df.loc[0] = scaler.transform(df)[0]
    prediction = best_model.predict_proba(df).flatten()
    data = {
        'prob': prediction.tolist()[1],
        'No': len(st.session_state['patients']) + 1,
        'arg': {key:st.session_state[key] for key in input_keys},
        'outcome': 'Yes' if prediction.tolist()[1] > 0.5 else 'No'
    }
    st.session_state['patients'].append(
        data
    )
    print('update patients ... ##########')

def plot_below_header():
    col1, col2 = st.columns([1, 9])
    col3, col4, col5, col6, col7 = st.columns([2, 2, 2, 2, 2])
    with col1:
        for i in range(8):
            st.write('')
        st.session_state['display'] = ['Single', 'Multiple'].index(
            st.radio("Display", ('Single', 'Multiple'), st.session_state['display']))
    with col2:
        plot_prediction()
    with col4:
        st.metric(
            label='Estimated probability',
            value="{:.2f}%".format(st.session_state['patients'][-1]['prob'] * 100)
        )

    with col6:
        st.metric(
            label='Estimated prediction',
            value=st.session_state['patients'][-1]['outcome']
        )
    st.write('')
    st.write('')
    st.write('')
    plot_patients()
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')

st.header('Random forest-based model for predicting postoperative complications', anchor='Postoperative complications')
if st.session_state['patients']:
    plot_below_header()
st.subheader("Instructions:")
st.write("1. Select patient's infomation on the left\n2. Press predict button\n3. The model will generate predictions")
st.write('***Note: this model is still a research subject, and the accuracy of the results cannot be guaranteed!***')
st.write("***[Paper link](https://pubmed.ncbi.nlm.nih.gov/)(To be updated)***")
with st.sidebar:
    with st.form("my_form",clear_on_submit = False):
        for code in sidebar_code:
            #print(code)
            exec(code)
        col8, col9, col10 = st.columns([3, 4, 3])
        with col9:
            prediction = st.form_submit_button(
                'Predict',
                on_click=predict,
                # args=[{key: eval(key.replace(' ', '____')) for key in input_keys}]
            )

