# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# Modified from https://github.com/hreikin/streamlit-uploads-library/blob/main/streamlit_uploads_library/library.py
# --------------------------------------------------------

import logging
from math import ceil

import streamlit as st

logger = logging.getLogger(__name__)


def is_from_pil(obj):
    # Get the name of the module where the class of obj is defined
    class_module = obj.__class__.__module__
    # Check if the module name starts with 'PIL.'
    return class_module.startswith('PIL.')


class Library():
    """Create a simple library out of streamlit widgets.

    Using the library is simple, import `streamlit_uploads_library` and then instantiate the class with the
    required `directory` variable. Other options can be configured by passing in different variables
    when instantiating the class.

    Example Usage:
        python
        import streamlit as st
        from library import Library

        st.set_page_config(page_title="Streamlit Uploads Library", layout="wide")
        default_library = Library(images=pil_images)
    """

    def __init__(self, media, media_alignment='end', number_of_columns=5):
        self.media = media
        self.media_alignment = media_alignment
        self.number_of_columns = number_of_columns
        self.root_container = self.create(media=self.media,
                                          media_alignment=self.media_alignment,
                                          number_of_columns=self.number_of_columns)

    def create(_self, media, media_alignment, number_of_columns):
        root_container = st.container()
        with root_container:
            col_idx = 0
            media_idx = 0
            max_idx = number_of_columns - 1
            num_of_files = len(media)
            num_of_rows_req = ceil(num_of_files / number_of_columns)
            library_rows = [st.container() for _ in range(num_of_rows_req)]
            library_rows_idx = 0

            for idx in range(num_of_rows_req):
                with library_rows[library_rows_idx]:
                    media_columns = list(st.columns(number_of_columns))

                for item in media[media_idx:(media_idx + number_of_columns)]:
                    with media_columns[col_idx]:
                        if is_from_pil(item):
                            st.image(item, use_column_width='auto')
                        elif isinstance(item, tuple):
                            if item[0] == 'image':
                                st.image(item[1], use_column_width='auto')
                            elif item[0] == 'video':
                                st.video(item[1])
                        elif isinstance(item, str):
                            st.video(item)
                        st.write(
                            f"""<style>
                                [data-testid="stHorizontalBlock"] {{
                                    align-items: {media_alignment};
                                }}
                                </style>
                                """,
                            unsafe_allow_html=True
                        )

                    if col_idx < max_idx:
                        col_idx += 1
                    else:
                        col_idx = 0
                        library_rows_idx += 1
                    media_idx += 1

        return root_container
