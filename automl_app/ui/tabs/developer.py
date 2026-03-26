import streamlit as st


def render_developer_tab() -> None:
    """Render the Developer Info tab with introduction and social media links."""
    
    st.markdown(
        """
        <style>
            .dev-hero {
                background: linear-gradient(135deg, #0f2233 0%, #214765 100%);
                border-radius: 20px;
                padding: 2.5rem 3rem;
                margin-bottom: 2rem;
                border: 1px solid rgba(255, 255, 255, 0.15);
                text-align: center;
                position: relative;
                overflow: hidden;
            }

            .dev-hero::before {
                content: '';
                position: absolute;
                top: -50%;
                right: -10%;
                width: 300px;
                height: 300px;
                background: radial-gradient(circle, rgba(22, 179, 160, 0.15) 0%, transparent 70%);
                border-radius: 50%;
            }

            .dev-avatar {
                font-size: 5rem;
                margin-bottom: 1rem;
                animation: float 3s ease-in-out infinite;
                position: relative;
                z-index: 1;
            }

            @keyframes float {
                0%, 100% { transform: translateY(0px); }
                50% { transform: translateY(-10px); }
            }

            .dev-name {
                font-size: 2.5rem;
                font-weight: 800;
                color: #f2f8ff;
                margin: 0.5rem 0;
                letter-spacing: -0.02em;
                font-family: 'Manrope', sans-serif;
                position: relative;
                z-index: 1;
            }

            .dev-title {
                font-size: 1.2rem;
                color: #16b3a0;
                font-weight: 700;
                margin-bottom: 1rem;
                position: relative;
                z-index: 1;
            }

            .dev-subtitle {
                font-size: 1rem;
                color: #d2e5f7;
                margin-bottom: 0;
                max-width: 700px;
                margin-left: auto;
                margin-right: auto;
                line-height: 1.6;
                position: relative;
                z-index: 1;
            }

            .social-links {
                display: flex;
                justify-content: center;
                gap: 1.5rem;
                margin: 2.5rem 0 0;
                flex-wrap: wrap;
                position: relative;
                z-index: 1;
            }

            .social-button {
                display: inline-flex;
                align-items: center;
                gap: 0.6rem;
                padding: 0.8rem 1.6rem;
                background: rgba(22, 179, 160, 0.15);
                border: 2px solid #16b3a0;
                border-radius: 12px;
                color: #16b3a0;
                text-decoration: none;
                font-weight: 700;
                font-size: 0.95rem;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                cursor: pointer;
            }

            .social-button:hover {
                background: linear-gradient(135deg, #16b3a0 0%, #0f8e7f 100%);
                color: #f2f8ff;
                transform: translateY(-3px);
                box-shadow: 0 12px 32px rgba(22, 179, 160, 0.3);
            }

            .info-card {
                background: #132231;
                border: 1px solid #2b4257;
                border-radius: 14px;
                padding: 1.8rem;
                box-shadow: 0 10px 28px rgba(10, 28, 45, 0.1);
                transition: all 0.3s ease;
            }

            .info-card:hover {
                border-color: #16b3a0;
                box-shadow: 0 14px 36px rgba(22, 179, 160, 0.15);
                transform: translateY(-4px);
            }

            .info-icon {
                font-size: 2.5rem;
                margin-bottom: 0.8rem;
            }

            .info-title {
                font-size: 1.1rem;
                font-weight: 700;
                color: #e8f2ff;
                margin-bottom: 0.6rem;
                font-family: 'Manrope', sans-serif;
            }

            .info-text {
                color: #a9bfd6;
                font-size: 0.95rem;
                line-height: 1.6;
            }

            .divider {
                height: 2px;
                background: linear-gradient(90deg, transparent, #2b4257, transparent);
                margin: 2.5rem 0;
                border: none;
            }

            .cta-section {
                background: linear-gradient(135deg, rgba(22, 179, 160, 0.1) 0%, rgba(15, 142, 127, 0.05) 100%);
                border: 1px solid rgba(22, 179, 160, 0.2);
                border-radius: 16px;
                padding: 2rem;
                text-align: center;
                margin-top: 3rem;
            }

            .cta-title {
                font-size: 1.4rem;
                font-weight: 700;
                color: #e8f2ff;
                margin-bottom: 0.8rem;
            }

            .cta-text {
                color: #a9bfd6;
                font-size: 1rem;
                margin-bottom: 1.5rem;
                line-height: 1.6;
            }

            .timeline {
                position: relative;
                padding: 2rem 0;
            }

            .timeline-item {
                padding-left: 3.5rem;
                padding-bottom: 2rem;
                position: relative;
            }

            .timeline-item::before {
                content: '';
                position: absolute;
                left: 0;
                top: 0;
                width: 12px;
                height: 12px;
                background: linear-gradient(135deg, #16b3a0 0%, #0f8e7f 100%);
                border-radius: 50%;
                border: 3px solid #132231;
            }

            .timeline-item::after {
                content: '';
                position: absolute;
                left: 4px;
                top: 20px;
                width: 2px;
                height: 100%;
                background: linear-gradient(180deg, #16b3a0 0%, transparent 100%);
            }

            .timeline-item:last-child::after {
                display: none;
            }

            .timeline-date {
                font-weight: 700;
                color: #16b3a0;
                font-size: 0.9rem;
                text-transform: uppercase;
            }

            .timeline-title {
                font-size: 1.1rem;
                font-weight: 700;
                color: #e8f2ff;
                margin: 0.4rem 0;
            }

            .timeline-desc {
                color: #a9bfd6;
                font-size: 0.95rem;
            }

            .stat-box {
                text-align: center;
                padding: 1.2rem;
                background: rgba(22, 179, 160, 0.05);
                border-radius: 12px;
                border: 1px solid rgba(22, 179, 160, 0.2);
            }

            .stat-number {
                font-size: 2rem;
                font-weight: 800;
                color: #16b3a0;
            }

            .stat-label {
                color: #a9bfd6;
                font-size: 0.9rem;
                margin-top: 0.4rem;
            }
        </style>

        <div class="dev-hero">
            <div class="dev-avatar">👨‍💻</div>
            <div class="dev-name">Himanshu Kumar</div>
            <div class="dev-title">Computer Science Engineering Student | AI Systems Builder</div>
            <div class="dev-subtitle">
                Focused on building real-world AI systems in Machine Learning and Generative AI.
                I learn by building, ship practical tools, and share everything through open source.
            </div>
            <div class="social-links">
                <a href="https://www.linkedin.com/in/himanshu231204/" target="_blank" class="social-button">
                    💼 LinkedIn
                </a>
                <a href="https://github.com/himanshu231204" target="_blank" class="social-button">
                    🐙 GitHub
                </a>
                <a href="https://x.com/himanshu231204" target="_blank" class="social-button">
                    🐦 Twitter/X
                </a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # About Section
    st.markdown("### 🌟 About Me")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(
            """
            I am a Computer Science Engineering student focused on building real-world AI systems
            in Machine Learning and Generative AI.

            I believe in learning by building, not just consuming tutorials, and sharing every
            project publicly through open source.
            
            **What I Do:**
            - Design AI systems, not just isolated models
            - Build retrieval, training, and deployment pipelines
            - Develop AI-powered developer tools
            - Create practical products for real workflows
            - Build in public and collaborate through open source
            """
        )
    
    with col2:
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown(
                """
                <div class="stat-box">
                    <div class="stat-number">3+</div>
                    <div class="stat-label">Flagship AI Projects</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_b:
            st.markdown(
                """
                <div class="stat-box">
                    <div class="stat-number">Open</div>
                    <div class="stat-label">Source First</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_c:
            st.markdown(
                """
                <div class="stat-box">
                    <div class="stat-number">Build</div>
                    <div class="stat-label">In Public</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # Skills Section
    st.markdown("### 💼 Core Competencies")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            """
            **Machine Learning**
            - ML model training and evaluation
            - Classification and regression workflows
            - Feature engineering and preprocessing
            - Performance benchmarking
            """
        )
    
    with col2:
        st.markdown(
            """
            **Generative AI and LLM Apps**
            - RAG application design
            - Retrieval with FAISS and LangChain
            - Explainable answer generation
            - Prompt and pipeline orchestration
            """
        )
    
    with col3:
        st.markdown(
            """
            **AI Developer Tooling and Systems**
            - AI-powered CLI tools
            - Backend systems for ML products
            - FastAPI and Streamlit applications
            - CI-ready engineering workflows
            """
        )

    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""**Tech Stack:** Python • C++ • LangChain • FAISS""")
    with col2:
        st.markdown("""**ML and App Stack:** Scikit-learn • Streamlit • FastAPI""")
    with col3:
        st.markdown("""**DevOps:** Docker • GitHub Actions • Git""")

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # Projects Section
    st.markdown("### 🚀 Featured Projects")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            """
            <div class="info-card">
                <div class="info-icon">📚</div>
                <div class="info-title">RAG AI for Document Q&A</div>
                <div class="info-text">
                    Built a retrieval-augmented generation application for document question answering
                    with explainable retrieval so users can trace answers back to source chunks.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.link_button(
            "🔗 View RAGNova Repo",
            "https://github.com/himanshu231204/ragnova-rag-chatbot",
            use_container_width=True,
        )
    
    with col2:
        st.markdown(
            """
            <div class="info-card">
                <div class="info-icon">🧠</div>
                <div class="info-title">AI Commit (Offline CLI)</div>
                <div class="info-text">
                    Developed a privacy-first CLI tool that generates intelligent Git commit messages
                    using local LLMs, designed for offline developer workflows.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.link_button(
            "🔗 View AI Commit Repo",
            "https://github.com/himanshu231204/ai-commit",
            use_container_width=True,
        )

    with col3:
        st.markdown(
            """
            <div class="info-card">
                <div class="info-icon">⚙️</div>
                <div class="info-title">AutoML Studio</div>
                <div class="info-text">
                    Built an end-to-end AutoML platform that automates EDA, preprocessing,
                    training, and evaluation with practical deployment-ready outputs.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.link_button(
            "🔗 View AutoML Studio Repo",
            "https://github.com/himanshu231204/AutoML-Studio-Pro-",
            use_container_width=True,
        )

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # Journey Section
    st.markdown("### 🎯 Focus Areas")
    
    st.markdown(
        """
        <div class="timeline">
            <div class="timeline-item">
                <div class="timeline-date">Current</div>
                <div class="timeline-title">Machine Learning and Evaluation</div>
                <div class="timeline-desc">
                    Building reproducible ML workflows with clear evaluation, benchmarking,
                    and practical model-selection pipelines.
                </div>
            </div>
            <div class="timeline-item">
                <div class="timeline-date">Current</div>
                <div class="timeline-title">Generative AI and LLM Applications</div>
                <div class="timeline-desc">
                    Designing RAG pipelines and LLM-powered applications focused on retrieval,
                    explainability, and real utility.
                </div>
            </div>
            <div class="timeline-item">
                <div class="timeline-date">Current</div>
                <div class="timeline-title">AI-Powered Developer Tools and Backend Systems</div>
                <div class="timeline-desc">
                    Creating developer productivity tools and backend systems that make
                    AI workflows faster, safer, and easier to adopt.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # Info Cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(
            """
            <div class="info-card">
                <div class="info-icon">🎓</div>
                <div class="info-title">Education and Approach</div>
                <div class="info-text">
                    Computer Science Engineering student focused on applied AI systems.
                    Strong belief in learning by building and validating ideas through real products.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with col2:
        st.markdown(
            """
            <div class="info-card">
                <div class="info-icon">🌍</div>
                <div class="info-title">Open Source and Collaboration</div>
                <div class="info-text">
                    Actively sharing projects in public repositories and open to collaboration,
                    discussions, and building practical AI tools with the community.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # CTA Section
    st.markdown(
        """
        <div class="cta-section">
            <div class="cta-title">Open to AI and GenAI Internships</div>
            <div class="cta-text">
                Currently seeking AI, ML, and Generative AI internship opportunities where I can
                contribute to real-world systems and grow as an engineer.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.link_button(
            "💼 LinkedIn Profile",
            "https://www.linkedin.com/in/himanshu231204/",
            use_container_width=True
        )
    with col2:
        st.link_button(
            "🐙 GitHub Profile",
            "https://github.com/himanshu231204",
            use_container_width=True
        )
    with col3:
        st.link_button(
            "🐦 Twitter/X Profile",
            "https://x.com/himanshu231204",
            use_container_width=True
        )
