import streamlit as st
import matplotlib.pyplot as plt

from utils import *

from utils import _get_state


######################################## INITIAL CONFIG ##########################################

# configuration of the page
st.set_page_config(
    page_title="MEDTEXT NLP",
    page_icon="💭",
    layout="wide",
    initial_sidebar_state="expanded",
)


# configuration for the matplots
size = 15
params = {
    "legend.fontsize": "large",
    "figure.figsize": (20, 10),
    "axes.labelsize": 25,
    "axes.titlesize": 25,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.titlepad": 25,
}
plt.rcParams["font.sans-serif"] = ["Avenir", "sans-serif"]
plt.rcParams.update(params)


st.title("Análisis de texto médico")
st.subheader(
    "Este proyecto tiene como objetivo extraer y mostrar información útil encontrada en los comentarios médicos."
)
st.write(
    "En la columna de la izquierda podemos generar comentarios *de mentira*, debido a la carencia de bases de datos públicas que hay."
)
st.write(
    "En la de la derecha, podemos analizar dichos comentarios, o aquellos de los que dispongamos nosotros."
)
col1, col2 = st.beta_columns((1, 3))


######################################## PAGE DEFINITION ##########################################

# GENERATION COLUMN
def comment_generation(state):
    st.header("GENERAR COMENTARIOS")
    n_comments = st.number_input("Nº de comentarios a generar", 1, 5, 2, 1)

    if st.button("Generar comentarios"):

        with st.spinner("Cargando el modelo en memoria"):

            model, tokenizer = load_model()

        st.success("Modelo cargado!")

        coms = st.empty()
        coms.write("Generando comentarios... puede tomar un rato")
        progress = st.progress(0)
        progress_text = st.empty()

        generated_comments = GeneratedComment(
            generate(
                model.to("cpu"),
                tokenizer,
                "<|BOS|>",
                entry_count=n_comments,
                progress_bar=progress,
                progress_text=progress_text,
            )
        )

        progress.empty()
        progress_text.empty()
        coms.empty()

        generated_comments = [
            l.removeprefix("<|BOS|>").rstrip("<|EOS|>\n") for l in generated_comments
        ]

        state.generated_comments = generated_comments

    if state.generated_comments is not None:
        st.markdown("---")
        for c in state.generated_comments:
            st.markdown(justify_text(c), unsafe_allow_html=True)
            st.markdown("---")

        if st.button("Borrar comentarios"):
            state.clear()


# EVALUATION COLUMN
def comment_evaluation(state):
    st.header("EVALUAR COMENTARIOS")

    selection = st.selectbox(
        "Elige el modelo a utilizar:",
        ["en_core_med7_trf", "en_ner_bionlp13cg_md", "en_ner_bc5cdr_md"],
        1,
    )

    with st.spinner(f"Cargando **{selection}**..."):
        ner_tagger = load_ner_tagger(selection)

    # configure the entities parser colours
    col_dict = {}
    seven_colours = [
        "#e6194B",
        "#3cb44b",
        "#ffe119",
        "#ffd8b1",
        "#f58231",
        "#f032e6",
        "#42d4f4",
        "#ff0000",
        "#ff8700",
        "#ffd300",
        "#deff0a",
        "#a1ff0a",
        "#0aff99",
        "#0aefff",
        "#147df5",
        "#580aff",
        "#be0aff",
    ]

    for label, colour in zip(ner_tagger.pipe_labels["ner"], seven_colours):
        col_dict[label] = colour

    html_format_options = {"ents": ner_tagger.pipe_labels["ner"], "colors": col_dict}

    source_comment_choice = st.radio(
        "Elige de dónde cargar los comentarios",
        ["Generados", "Desde archivo", "Escribir"],
        index=0,
    )

    texts = None
    if source_comment_choice == "Generados":
        if state.generated_comments is not None:
            texts = state.generated_comments
        else:
            st.warning("Debes generar comentarios primero!")

    elif source_comment_choice == "Desde archivo":
        st.warning(
            "Sube un archivo de texto con tus comentarios aquí. Debe ser un .txt o .dat. Se espera que haya un comentario por línea."
        )
        uploaded_file = st.file_uploader("", type=["txt", "dat"])
        if uploaded_file is not None:
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            texts = stringio.readlines()

    elif source_comment_choice == "Escribir":
        st.subheader(
            "Puedes escribir tus comentarios aquí. Escribe un solo comentario por línea."
        )
        texts = st.text_area("Escribe aquí tus comentarios.", height=200).split("\n")

    if texts is None:
        pass

    else:

        if len(texts) == 1:
            state.text_idx = 0

        else:
            state.text_idx = st.selectbox(
                "Elige el comentario a analizar", options=list(range(len(texts)))
            )

        state.text = load_comment(texts[state.text_idx])

        doc = ner_tagger(state.text)

        styled_html = spacy.displacy.render(
            doc, style="ent", options=html_format_options
        )

        styled_html = check_colors_html(styled_html)

        st.markdown(styled_html, unsafe_allow_html=True)

        st.markdown("---")

        display_analysis(state.text, doc, col_dict)


def main():
    state = _get_state()
    pages = {
        "Generation": comment_generation,
        "Evaluation": comment_evaluation,
    }

    # Display the selected page with the session state
    with col1:
        pages["Generation"](state)

    with col2:
        pages["Evaluation"](state)

    # Mandatory to avoid rollbacks with widgets, must be called at the end of your app
    state.sync()


if __name__ == "__main__":
    main()
