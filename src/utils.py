from dataclasses import dataclass, field
from genericpath import exists
from io import StringIO
from collections import Counter

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from colour import Color

import spacy
from transformers import GPT2LMHeadModel, GPT2Tokenizer

import torch
import torch.nn.functional as F
import streamlit as st
from streamlit.hashing import _CodeHasher
from streamlit.report_thread import get_report_ctx
from streamlit.server.server import Server

import requests

MODEL_ID_GD = "1KGYphUBa8CttAAdt88GfXRYxYn-J5Kqm"


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={"id": id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


class _SessionState:
    def __init__(self, session, hash_funcs):
        """Initialize SessionState instance."""
        self.__dict__["_state"] = {
            "data": {},
            "hash": None,
            "hasher": _CodeHasher(hash_funcs),
            "is_rerun": False,
            "session": session,
        }

    def __call__(self, **kwargs):
        """Initialize state data once."""
        for item, value in kwargs.items():
            if item not in self._state["data"]:
                self._state["data"][item] = value

    def __getitem__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)

    def __getattr__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)

    def __setitem__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value

    def __setattr__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value

    def clear(self):
        """Clear session state and request a rerun."""
        self._state["data"].clear()
        self._state["session"].request_rerun()

    def sync(self):
        """Rerun the app with all state values up to date from the beginning to fix rollbacks."""

        # Ensure to rerun only once to avoid infinite loops
        # caused by a constantly changing state value at each run.
        #
        # Example: state.value += 1
        if self._state["is_rerun"]:
            self._state["is_rerun"] = False

        elif self._state["hash"] is not None:
            if self._state["hash"] != self._state["hasher"].to_bytes(
                self._state["data"], None
            ):
                self._state["is_rerun"] = True
                self._state["session"].request_rerun()

        self._state["hash"] = self._state["hasher"].to_bytes(self._state["data"], None)


def _get_state(hash_funcs=None):
    session = _get_session()

    if not hasattr(session, "_custom_session_state"):
        session._custom_session_state = _SessionState(session, hash_funcs)

    return session._custom_session_state


def _get_session():
    session_id = get_report_ctx().session_id
    session_info = Server.get_current()._get_session_info(session_id)

    if session_info is None:
        raise RuntimeError("Couldn't get your Streamlit Session object.")

    return session_info.session


@st.cache
class GeneratedComment:
    def __init__(self, comments):
        self.comment_list = comments

    def __getitem__(self, item):
        return self.comment_list[item]


def normalize(n, n_min=0, n_max=1):
    return (n - n_min) / (n_max - n_min)


@st.cache(allow_output_mutation=True, show_spinner=False)
def load_model():
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    if not exists("trained_models/medtext-final.pt"):
        os.mkdir("trained_models")
        download_file_from_google_drive(MODEL_ID_GD, "trained_models/medtext-final.pt")

    model.load_state_dict(
        torch.load("trained_models/medtext-final.pt", map_location=torch.device("cpu"))
    )
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    return model, tokenizer


@st.cache(allow_output_mutation=True, show_spinner=False)
def load_ner_tagger(tagger):
    return spacy.load(tagger)


@st.cache(allow_output_mutation=True)
def load_comment(text):
    return text


def b_or_w_font(hex_value):
    color = Color(f"#{hex_value}")

    if color.get_luminance() > 0.45:
        value = "#000000"
    else:
        value = "#ffffff"
    return value


def check_colors_html(styled_html):
    color_list = re.findall(r"(?<=background: #)\w+", styled_html)
    text_color_list = [b_or_w_font(c) for c in color_list]

    for c, t in zip(color_list, text_color_list):
        styled_html = re.sub(f"#{c};", f"#{c}; color: {t};", styled_html)

    return styled_html


def justify_text(text):
    return f'<div style="text-align: justify"> {text} </div>'


def generate(
    model,
    tokenizer,
    prompt,
    entry_count=10,
    entry_length=100,
    top_p=0.8,
    temperature=1.0,
    progress_bar=None,
    progress_text=None,
):

    model.eval()

    generated_num = 0
    generated_list = []

    filter_value = -float("Inf")

    with torch.no_grad():

        for entry_idx in range(entry_count):
            current_stop = entry_idx

            entry_finished = False

            generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)

            # Using top-p (nucleus sampling): https://github.com/huggingface/transformers/blob/master/examples/run_generation.py

            progress_text.write(f"{entry_idx + 1} / {entry_count} comentarios...")
            for i in range(entry_length):

                current_word = normalize(
                    current_stop + i / entry_length,
                    n_max=entry_count,
                )

                progress_bar.progress(current_word)

                outputs = model(generated, labels=generated)
                loss, logits = outputs[:2]
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)

                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value

                next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
                generated = torch.cat((generated, next_token), dim=1)

                if next_token in tokenizer.encode("<|EOS|>"):
                    entry_finished = True

                if entry_finished:

                    generated_num = generated_num + 1

                    output_list = list(generated.squeeze().numpy())
                    output_text = tokenizer.decode(output_list)

                    generated_list.append(output_text)
                    break

            if not entry_finished:
                output_list = list(generated.squeeze().numpy())
                output_text = f"{tokenizer.decode(output_list)}<|EOS|>"
                generated_list.append(output_text)

    return generated_list


def display_analysis(text, doc, col_dict):
    with st.beta_expander("Mostrar más datos", expanded=True):
        st.write("### Algunas gráficas y estadísticas de los comentarios...")

        num_tokens = len(text.split(" "))
        num_chars = len(text)

        st.markdown("# Tamaño de nuestro comentario:")

        st.markdown(f"\t### {num_tokens} tokens")
        st.markdown(f"\t### {num_chars} caracteres")

        st.markdown("---")

        text_data = [(ent.text, ent.label_) for ent in doc.ents]
        counter = Counter([t[1] for t in text_data])

        if len(counter.items()) > 0:
            st.markdown("# Se encontraron las siguientes etiquetas...")
            for k, v in counter.items():
                st.markdown(f"### {v} {k}")

            if len(counter.items()) > 1:
                df = pd.DataFrame(counter.items(), columns=["Tag", "Count"])

                fig = plt.figure(figsize=(5, 2))
                plt.barh(df.Tag, df.Count, color=[col_dict[tag] for tag in df.Tag])
                st.pyplot(fig)
        else:
            st.markdown("## El modelo no encontró ninguna etiqueta.")
            st.markdown(
                "#### Puedes probar a cambiar el modelo arriba, ofrecerá otros resultados."
            )
            st.markdown("---")
