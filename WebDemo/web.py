from logging import PlaceHolder
from re import sub
import streamlit as st
import imp, time, random
import base64
import io
import nbformat
from PIL import Image
from datasets import load_from_disk
import os
from Ensembleapi import Ensemble
os.environ["TOKENIZERS_PARALLELISM"] = "false"

st.set_page_config(layout="wide")

def set_submitted_true():
    st.session_state.submitted=True

st.markdown("""
<style>
input, .rtl {
  unicode-bidi:bidi-override;
  direction: RTL;
}
textarea, .rtl {
  unicode-bidi:bidi-override;
  direction: RTL;
}
h2, .rtl {
  unicode-bidi:bidi-override;
  direction: RTL;
}
div[role=tablist], .rtl {
  unicode-bidi:bidi-override;
  direction: RTL;
}
div[role=alert], .rtl {
  unicode-bidi:bidi-override;
  direction: RTL;
}
</style>
    """, unsafe_allow_html=True)


latest_iteration = st.empty()
bar = st.progress(0)

@st.cache(allow_output_mutation=True)
def load_models(models):
    predictors = []
    for idx, model in enumerate(models):
        latest_iteration.text(f'Loading {model}')
        bar.progress((idx + 1)/len(models))
        module = imp.load_source(model, "./WebDemo/"+model+"api.py")
        modelClass = getattr(module, model)
        predictors.append(modelClass())
    return predictors

@st.cache(allow_output_mutation=True)
def load_dataset():
    dataset = load_from_disk("./Datasets/test.hf").shuffle(seed=42)
    return dataset

models = ["ParsBERT", "ALBERT", "MBERT"]#, "ParsT5"]

predictors = load_models(models)
bar.empty()

models += ["Ensemble"]
predictors.append( Ensemble(predictors) )

dataset = load_dataset()

st.markdown("## 📝 مدل های پرسش و پاسخ فارسی")
# with st.container():

#     st.write(
#         """
# -   دموی چندین مدل پرسش و پاسخ فارسی در این اپلیکیشن فراهم شده است که از منوی سمت چپ قابل انتخاب هستند..\n
# -   همچنین قابلیت مقایسه خروجی مدل ها،!.\n
# -   You can also compare models by compare checkbox in sidebar. Code of models are available in code tab.
# 	    """
#     )

st.markdown("")

tab1, tab2 = st.tabs(["دموی مدل", "نوتبوک آموزش مدل"])

selected_model = st.sidebar.radio(
    'Which model do you like to try?',
     models)
selected_model_idx = models.index(selected_model)


cmp_model_idx = selected_model_idx
cmp_model = st.sidebar.selectbox(
    f"Which model do you like to compare {models[selected_model_idx]} with?",
    ["None"]+[model for model in models if model!=models[selected_model_idx]],
    on_change = set_submitted_true)

do_compare=(cmp_model!="None")
if not do_compare:
    cmp_model_idx=0
else:
    cmp_model_idx = models.index(cmp_model)

st.sidebar.info("تمامی دادگان، کد ها و نتایج ارزیابی مدل ها در [صفحه گیت هاب پروژه](https://github.com/AmirMansurian/PQA) قابل دسترسی است", icon="ℹ️")

with tab1.form("my_form", clear_on_submit=False):

    col1, col2, col3 = st.columns(3)
    with col3:
        st.caption('می توانید با فشردن دکمه رو به رو یک داده تصادفی ایجاد کنید')
    with col1:
        generate_random_data = st.form_submit_button("تولید تصادفی داده")
        if generate_random_data:
            sample_idx = random.randrange(len(dataset))
            st.session_state.context = dataset[sample_idx]["context"]
            st.session_state.question = dataset[sample_idx]["question"]

    if 'context' in st.session_state and st.session_state.context is not None:
        context = st.text_area(label="Context", key="context", height=300, value=st.session_state.context)
        question = st.text_input(label="Question", key="question", value=st.session_state.question)
    else:
        context = st.text_area(label="Context", height=300, placeholder="متن مورد نظر را اینجا وارد کنید ...")
        question = st.text_input(label="Question", placeholder="سوال خود از متن را اینجا بپرسید  ...")

    submitted = st.form_submit_button("Submit")
    if submitted or ('submitted' in st.session_state and st.session_state.submitted):
        st.session_state.submitted = False
        selected_prediction = predictors[selected_model_idx](question, context)[0]
        cmp_prediction = predictors[cmp_model_idx](question, context)[0]
        if do_compare:
            col1, col2 = st.columns(2)

            with col1:
                st.text_area(label=f"{models[selected_model_idx]}'s Answer:", value=(selected_prediction if selected_prediction!="" else "بدون پاسخ"))

            with col2:
                st.text_area(label=f"{models[cmp_model_idx]}'s Answer:", value=(cmp_prediction if cmp_prediction!="" else "بدون پاسخ"))
        else:
            st.text_area(label=f"{models[selected_model_idx]}'s Answer:", value=selected_prediction)#(selected_prediction if selected_prediction!="" else "بدون پاسخ"))



############ Code tab ###############

notebooks = ["./Models/ParsBert.ipynb", "./Models/albert.ipynb", "./Models/mbert.ipynb", "./Models/ensemble.ipynb"] # "./Models/ParsT5.ipynb" before ensemble

nb: nbformat.notebooknode.NotebookNode = nbformat.read(notebooks[selected_model_idx], as_version=4)

for cell in nb.cells:
    cell_container = tab2.container()

    left_col, right_col = cell_container.columns((3, 1))

    with left_col:
        if cell["cell_type"] == "markdown":
            tab2.markdown(cell["source"])
        elif cell["cell_type"] == "code":
            tab2.code(cell["source"])

    if "outputs" not in cell:
        continue

    with right_col:
        for output in cell["outputs"]:
            if output["output_type"] == "stream":
                tab2.text(output["text"])
            elif output["output_type"] == "display_data" or output["output_type"] == "execute_result":
                output_data = output["data"]
                if "text/plain" in output_data:
                    tab2.text(output_data["text/plain"])
                if "image/png" in output_data:
                    image_data = base64.b64decode(output_data["image/png"])
                    tab2.image(Image.open(io.BytesIO(image_data)))
                if "application/json" in output_data:
                    tab2.json(output_data["application/json"])
            elif output["output_type"] == "error":
                tab2.error(f'{output["ename"]} {output["evalue"]} /n {"/n".join(output["traceback"])}')