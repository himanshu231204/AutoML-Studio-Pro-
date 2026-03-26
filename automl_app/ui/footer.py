import streamlit as st


def render_footer() -> None:
    st.divider()
    st.markdown(
        """
        <div style="text-align:center; font-size:14px;">
            👨‍💻 Developed by <b>Himanshu Kumar</b><br><br>
            🔗
            <a href="https://www.linkedin.com/in/himanshu231204" target="_blank">LinkedIn</a> |
            <a href="https://github.com/himanshu231204" target="_blank">GitHub</a>
        </div>
        """,
        unsafe_allow_html=True,
    )
