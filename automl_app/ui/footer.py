import streamlit as st


def render_footer() -> None:
    st.markdown(
        """
        <style>
            .footer-container {
                margin-top: 3rem;
                padding: 2.5rem 1.5rem;
                text-align: center;
            }

            .footer-divider {
                height: 2px;
                background: linear-gradient(90deg, transparent, var(--line), transparent);
                margin: 2rem 0;
                border: none;
            }

            .footer-content {
                font-size: 14px;
                color: var(--muted);
            }

            .footer-author {
                font-weight: 700;
                color: var(--ink);
                font-size: 15px;
                margin-bottom: 0.8rem;
            }

            .footer-links {
                display: flex;
                justify-content: center;
                gap: 2rem;
                margin-top: 1rem;
                flex-wrap: wrap;
            }

            .footer-links a {
                color: var(--accent);
                text-decoration: none;
                font-weight: 600;
                transition: all 0.25s ease;
                padding: 0.4rem 0.8rem;
                border-radius: 8px;
            }

            .footer-links a:hover {
                color: var(--hero-text);
                background: rgba(22, 179, 160, 0.15);
                transform: translateY(-2px);
            }

            .footer-badge {
                display: inline-block;
                padding: 0.4rem 0.8rem;
                background: rgba(22, 179, 160, 0.1);
                border: 1px solid var(--accent);
                border-radius: 20px;
                color: var(--accent);
                font-size: 12px;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                margin-top: 1rem;
            }
        </style>
        
        <div class="footer-container">
            <div class="footer-divider"></div>
            <div class="footer-content">
                <div class="footer-author">👨‍💻 Developed by Himanshu Kumar</div>
                <div style="font-size: 13px; margin-bottom: 1rem;">Intelligent ML Model Training & Deployment Platform</div>
                <div class="footer-links">
                    <a href="https://www.linkedin.com/in/himanshu231204/" target="_blank" rel="noopener noreferrer">
                        💼 LinkedIn
                    </a>
                    <a href="https://github.com/himanshu231204" target="_blank" rel="noopener noreferrer">
                        🐙 GitHub
                    </a>
                    <a href="https://x.com/himanshu231204" target="_blank" rel="noopener noreferrer">
                        🐦 Twitter/X
                    </a>
                </div>
                <div class="footer-badge">✨ Production Ready v1.0</div>
                <div style="margin-top: 1.5rem; font-size: 12px; color: var(--muted); opacity: 0.8;">
                    © 2026 AutoML Studio Pro. All rights reserved.<br>
                    Built with ❤️ using Streamlit, Scikit-Learn & Python
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
