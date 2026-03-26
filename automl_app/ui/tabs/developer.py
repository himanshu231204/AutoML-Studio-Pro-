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
            <div class="dev-title">Full-Stack ML Engineer & Data Scientist</div>
            <div class="dev-subtitle">
                Passionate about building intelligent systems that solve real-world problems. 
                Specialized in AutoML, production ML pipelines, and data-driven decision making.
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
            I'm a passionate full-stack ML engineer with expertise in building end-to-end 
            machine learning solutions. My journey in tech started with a curiosity about 
            how data can drive intelligent decisions, and it evolved into a professional 
            mission to democratize machine learning.
            
            **What I Do:**
            - Design and deploy scalable ML systems
            - Automate machine learning workflows
            - Build production-ready data pipelines
            - Create intelligent web applications
            - Mentor and teach ML concepts
            """
        )
    
    with col2:
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown(
                """
                <div class="stat-box">
                    <div class="stat-number">10+</div>
                    <div class="stat-label">ML Projects</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_b:
            st.markdown(
                """
                <div class="stat-box">
                    <div class="stat-number">100%</div>
                    <div class="stat-label">Dedication</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_c:
            st.markdown(
                """
                <div class="stat-box">
                    <div class="stat-number">5+</div>
                    <div class="stat-label">Years Experience</div>
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
            - Scikit-Learn, TensorFlow
            - Classification & Regression
            - Feature Engineering
            - Model Optimization
            """
        )
    
    with col2:
        st.markdown(
            """
            **Data Engineering**
            - Pandas, NumPy, SQL
            - Data Pipelines
            - ETL Processing
            - Data Validation
            """
        )
    
    with col3:
        st.markdown(
            """
            **Web Development**
            - Streamlit, Python
            - FastAPI, Flask
            - Full-Stack Integration
            - UI/UX Design
            """
        )

    st.markdown()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""**Technologies:** Python, SQL, Git""")
    with col2:
        st.markdown("""**Tools:** Jupyter, VS Code, Docker""")
    with col3:
        st.markdown("""**Platforms:** AWS, GitHub, Streamlit Cloud""")

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # Projects Section
    st.markdown("### 🚀 Featured Projects")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(
            """
            <div class="info-card">
                <div class="info-icon">🎓</div>
                <div class="info-title">AutoML Studio Pro</div>
                <div class="info-text">
                    Enterprise-grade ML platform automating model training, evaluation, 
                    and deployment. Features include advanced EDA, multi-metric optimization, 
                    and production-ready artifact generation.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with col2:
        st.markdown(
            """
            <div class="info-card">
                <div class="info-icon">📊</div>
                <div class="info-title">Data Analytics Suite</div>
                <div class="info-text">
                    Comprehensive analytics platform with real-time dashboards, 
                    advanced visualizations, and predictive analytics. Deployed 
                    across multiple organizations.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # Journey Section
    st.markdown("### 🎯 Professional Journey")
    
    st.markdown(
        """
        <div class="timeline">
            <div class="timeline-item">
                <div class="timeline-date">2020 - Present</div>
                <div class="timeline-title">Full-Stack ML Engineer</div>
                <div class="timeline-desc">
                    Designing and deploying end-to-end machine learning solutions. 
                    Specialized in AutoML platforms and production ML pipelines.
                </div>
            </div>
            <div class="timeline-item">
                <div class="timeline-date">2018 - 2020</div>
                <div class="timeline-title">Data Scientist</div>
                <div class="timeline-desc">
                    Built predictive models and analytics dashboards. 
                    Led data-driven decision making initiatives.
                </div>
            </div>
            <div class="timeline-item">
                <div class="timeline-date">2016 - 2018</div>
                <div class="timeline-title">Junior Developer</div>
                <div class="timeline-desc">
                    Started with Python and data analysis. Developed foundation in 
                    software engineering and best practices.
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
                <div class="info-title">Education</div>
                <div class="info-text">
                    Academic foundation in Computer Science with specialization 
                    in Machine Learning and Data Science. Continuous learner 
                    with certifications in advanced ML techniques.
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
                <div class="info-title">Open Source</div>
                <div class="info-text">
                    Active contributor to open-source ML projects. 
                    Passionate about sharing knowledge and building 
                    community tools that help others succeed.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # CTA Section
    st.markdown(
        """
        <div class="cta-section">
            <div class="cta-title">Let's Connect! 🤝</div>
            <div class="cta-text">
                I'm always excited to collaborate on interesting ML projects, discuss ML strategies, 
                or help with your data science challenges.
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
